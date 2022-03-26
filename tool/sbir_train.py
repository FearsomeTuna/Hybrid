import os
import random
from re import A
import time
import cv2
import numpy as np
import logging
import argparse
import shutil
import math

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter

from model.hybrid import MixedBiModal

from util import config
from util.util import AverageMeter, find_free_port, smooth_loss, cal_accuracy, ContrastiveLossHardNegative, combination_cosine_similarity
from util.dataset import SBIRDataset, TxtFileDataset
from util.sampler import DistributedEvalSampler

from torchmetrics import RetrievalMAP

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/imagenet/imagenet_san10_pairwise.yaml', help='config file')
    parser.add_argument('opts', help='see files in config folder for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args, best_mAP
    args, best_mAP = argss, 0
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    model = MixedBiModal(args.layers, args.layer_types, args.widths, args.classes, args.layer_types[0], args.share_layers,args.sa_type if 'san' in args.layer_types else None, args.added_nl_blocks)
    triplet_model = MixedBiModal(args.layers, args.layer_types, args.widths, args.classes, args.layer_types[0], args.share_layers,args.sa_type if 'san' in args.layer_types else None, args.added_nl_blocks)

    # contrastive training uses frozen weights for unshared portion
    for p in model.unsharedParameters():
        p.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info("Model total num of params: {}".format(total_params))
        logger.info("Model total num of trainable params (contrastive phase): {}".format(total_params_trainable))
        logger.info(model)

    # pretrained classification weights
    if args.sketch_weight and args.image_weight:
        if os.path.isfile(args.sketch_weight) and os.path.isfile(args.image_weight):
            if main_process():
                logger.info("=> loading unshared weights. Sketch weights: '{}'. Image weights: '{}'".format(args.sketch_weight, args.image_weight))
            checkpoint_sketch = torch.load(args.sketch_weight, map_location=lambda storage, loc: storage.cuda(gpu))
            checkpoint_image = torch.load(args.image_weight, map_location=lambda storage, loc: storage.cuda(gpu))
            # if state dict comes from DP or DDP model, remove prefix
            nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint_sketch['state_dict'], "module.")
            nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint_image['state_dict'], "module.")
            model.load_unshared_state_dict(checkpoint_sketch['state_dict'], checkpoint_image['state_dict'])
            if main_process():
                logger.info("=> loaded weights '{}' and '{}'".format(args.sketch_weight, args.image_weight))
        else:
            if main_process():
                logger.info("=> no weight found at '{}'".format(args.weight))


    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])
        triplet_model = torch.nn.parallel.DistributedDataParallel(triplet_model.cuda(), device_ids=[gpu])
    else:
        model = torch.nn.DataParallel(model.cuda())
        triplet_model = torch.nn.DataParallel(triplet_model.cuda())

    classif_criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    contras_criterion = ContrastiveLossHardNegative(args.margin)
    triplet_criterion = nn.TripletMarginLoss(margin=args.margin)

    contras_optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    triplet_optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    contras_scheduler = lr_scheduler.CosineAnnealingLR(contras_optimizer, T_max=args.contras_epochs)
    triplet_scheduler = lr_scheduler.CosineAnnealingLR(triplet_optimizer, T_max=args.triplet_epochs)

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight, map_location=lambda storage, loc: storage.cuda(gpu))
            model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            if main_process():
                logger.info("=> no weight found at '{}'".format(args.weight))
    
    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(gpu))
            args.start_epoch = checkpoint['epoch']
            best_mAP = checkpoint['mAP_val']
            if args.start_epoch < args.contras_epochs:
                model.load_state_dict(checkpoint['state_dict'])
                contras_optimizer.load_state_dict(checkpoint['contras_optimizer'])
                contras_scheduler.load_state_dict(checkpoint['contras_scheduler'])
            else:
                triplet_model.load_state_dict(checkpoint['state_dict'])
                triplet_optimizer.load_state_dict(checkpoint['triplet_optimizer'])
                triplet_scheduler.load_state_dict(checkpoint['triplet_scheduler'])
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    image_mean, image_std = args.image_mean, args.image_std
    sketch_mean, sketch_std = args.sketch_mean, args.sketch_std

    image_train_transform = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.7, 1)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(image_mean, image_std)])
    sketch_train_transform = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.9, 1)), transforms.RandomHorizontalFlip(), transforms.RandomRotation((-20,20), fill=255), transforms.ToTensor(), transforms.Normalize(sketch_mean, sketch_std)])

    image_val_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(image_mean, image_std)])
    sketch_val_transform = transforms.Compose([transforms.Resize(224, max_size=225), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(sketch_mean, sketch_std)])

    train_set = SBIRDataset(args.sketch_root, args.image_root, 'train.txt', 'train.txt', 'mapping.txt', 'mapping.txt', '\t', True, sketch_transform=sketch_train_transform, image_transform=image_train_transform)
    val_sketch_set = TxtFileDataset(args.sketch_root, 'val_queries.txt', 'mapping.txt', '\t', True, transform=sketch_val_transform, return_index=True)
    val_img_set = TxtFileDataset(args.image_root, 'val.txt', 'mapping.txt', '\t', True, transform=image_val_transform)

    assert len(val_sketch_set.classes) == len(val_img_set.classes)
    for i in range(len(val_sketch_set.classes)):
        if val_sketch_set.classes[i] != val_img_set.classes[i]:
            raise ValueError('Image and sketch datasets class mismatch. Image: {}, sketch: {}. Make sure class names match when making dataset_paths file to ensure same class indexing.'.format(val_img_set.classes[i], val_sketch_set.classes[i]))
    for category in val_sketch_set.classes:
        if val_sketch_set.class_to_idx[category] != val_img_set.class_to_idx[category]:
            raise ValueError('Index mismatch in image and sketch datasets. Class {} has index {} in image dataset and index {} in sketches. Make sure class names match when making dataset_paths file to ensure same class indexing.'.format(category, val_img_set.class_to_idx[category], val_sketch_set.class_to_idx[category]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        val_img_sampler = DistributedEvalSampler(val_img_set, shuffle=False)
        val_sketch_sampler = None
    else:
        train_sampler = None
        val_img_sampler = None
        val_sketch_sampler = None
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_img_loader = torch.utils.data.DataLoader(val_img_set, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_img_sampler)
    val_sketch_loader = torch.utils.data.DataLoader(val_sketch_set, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sketch_sampler)

    sum_epochs = args.contras_epochs + args.triplet_epochs

    for epoch in range(args.start_epoch, sum_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if epoch < args.contras_epochs: # contrastive training
            train_set.setMode(SBIRDataset.MODES.PAIR)
            loss_train = contras_train(train_loader, model, contras_criterion, contras_optimizer, epoch, classif_criterion)
            mAP_val = retrieval_validate(val_img_loader, val_sketch_loader, model, combination_cosine_similarity)
            contras_scheduler.step()
            epoch_log = epoch + 1
            if main_process():
                writer.add_scalar('loss_train', loss_train, epoch_log)
                writer.add_scalar('mAP_val', mAP_val, epoch_log)
        elif epoch >= args.contras_epochs: # triplet training
            # on first triplet training epoch, we should unfreeze layers. Since we can't change ddp model after initialization, so we make a new model
            if epoch == args.contras_epochs:
                if main_process():
                    logger.info("=> Starting triplet training phase. Transfering weights to triplet model with no frozen layers...")
                state_dict = model.state_dict()
                triplet_model.load_state_dict(state_dict)

            train_set.setMode(SBIRDataset.MODES.TRIPLET)
            loss_train = triplet_train(train_loader, triplet_model, triplet_criterion, triplet_optimizer, epoch, classif_criterion)
            mAP_val = retrieval_validate(val_img_loader, val_sketch_loader, triplet_model, combination_cosine_similarity)
            triplet_scheduler.step()
            epoch_log = epoch + 1
            if main_process():
                writer.add_scalar('loss_train', loss_train, epoch_log)
                writer.add_scalar('mAP_val', mAP_val, epoch_log)

        if (epoch_log % args.save_freq == 0) and main_process():
            save_model = model if epoch < args.contras_epochs else triplet_model
            filename = args.save_path + '/train_epoch_' + str(epoch_log) + '.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': save_model.state_dict(), 'contras_optimizer': contras_optimizer.state_dict(),
                        'contras_scheduler': contras_scheduler.state_dict(), 'triplet_optimizer': triplet_optimizer.state_dict(),
                        'triplet_scheduler': triplet_scheduler.state_dict(), 'mAP_val': mAP_val}, filename)
            if mAP_val > best_mAP:
                best_mAP = mAP_val
                shutil.copyfile(filename, args.save_path + '/model_best.pth')
            if epoch_log / args.save_freq > 2:
                deletename = args.save_path + '/train_epoch_' + str(epoch_log - args.save_freq * 2) + '.pth'
                os.remove(deletename)
    if main_process():
        writer.close()


def contras_train(train_loader, model, criterion, optimizer, epoch, classif_criterion):
    iter_time = AverageMeter()
    data_time = AverageMeter()
    transfer_time = AverageMeter()
    batch_time = AverageMeter()
    metric_time = AverageMeter()
    loss_meter = AverageMeter()
    loss_sketch_meter = AverageMeter()
    loss_contras_meter = AverageMeter()
    loss_image_meter = AverageMeter()
    top1_meter_sketch = AverageMeter()
    top1_meter_image = AverageMeter()

    model.train()
    end = time.time()
    sum_epochs = args.contras_epochs + args.triplet_epochs
    max_iter = sum_epochs * len(train_loader)
    for i, (sketch_input, image_input, sketch_target, image_target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if args.time_breakdown:
            last = time.time()

        sketch_input = sketch_input.cuda(non_blocking=True)
        image_input = image_input.cuda(non_blocking=True)
        sketch_target = sketch_target.cuda(non_blocking=True)
        image_target = image_target.cuda(non_blocking=True)
        if args.time_breakdown:
            torch.cuda.synchronize()
            transfer_time.update(time.time() - last)
            last = time.time()

        (sketchFeatVec, imageFeatVec), (sketch_output, image_output) = model(sketch_input, image_input)
        loss_sketch = smooth_loss(sketch_output, sketch_target, args.label_smoothing) if args.label_smoothing else classif_criterion(sketch_output, sketch_target)
        loss_image = smooth_loss(image_output, image_target, args.label_smoothing) if args.label_smoothing else classif_criterion(image_output, image_target)
        loss_contras = criterion(F.normalize(sketchFeatVec), F.normalize(imageFeatVec), sketch_target, image_target)

        loss_contras, loss_sketch, loss_image = loss_contras*2.0, loss_sketch*0.5, loss_image*0.5
        classif_decay= 0.5*(1+ math.cos(epoch*math.pi/args.contras_epochs))
        loss = loss_contras*epoch + classif_decay*(loss_sketch + loss_image)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.time_breakdown:
            torch.cuda.synchronize()
            batch_time.update(time.time() - last)
            last = time.time()

        n = sketch_input.size(0)
        if args.multiprocessing_distributed:
            with torch.no_grad():
                loss, loss_contras, loss_sketch, loss_image = loss.detach() * n, loss_contras.detach() * n, loss_sketch.detach() *n, loss_image.detach()*n
                count = sketch_target.new_tensor([n], dtype=torch.long)
                dist.all_reduce(loss), dist.all_reduce(count), dist.all_reduce(loss_contras), dist.all_reduce(loss_sketch), dist.all_reduce(loss_image)
                n = count.item()
                loss, loss_contras, loss_sketch, loss_image = loss / n, loss_contras/n, loss_sketch/n, loss_image/n
        loss_meter.update(loss.item(), n)
        loss_contras_meter.update(loss_contras.item(), n)
        loss_sketch_meter.update(loss_sketch.item(), n)
        loss_image_meter.update(loss_image.item(), n)

        # classification metrics
        top1_s, = cal_accuracy(sketch_output, sketch_target, topk=(1,))
        top1_i, = cal_accuracy(image_output, image_target, topk=(1,))
        n = sketch_input.size(0)
        if args.multiprocessing_distributed:
            with torch.no_grad():
                top1_s = top1_s * n
                top1_i = top1_i * n
                count = sketch_target.new_tensor([n], dtype=torch.long)
                dist.all_reduce(top1_s), dist.all_reduce(top1_i), dist.all_reduce(count)
                n = count.item()
                top1_s = top1_s / n
                top1_i = top1_i / n    
        top1_meter_sketch.update(top1_s.item(), n), top1_meter_image.update(top1_i.item(), n)            

        if args.time_breakdown:
            torch.cuda.synchronize()
            metric_time.update(time.time() - last)
        iter_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * iter_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if ((i + 1) % args.print_freq == 0) and main_process():
            logger.info(('Epoch: [{}/{}][{}/{}] '
                        'Time {iter_time.val:.3f} ({iter_time.avg:.3f}) '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) ' +
                        (
                            'Transfer {transfer_time.val:.3f} ({transfer_time.avg:.3f}) '
                            'Batch {batch_time.val:.4f} ({batch_time.avg:.4f}) '
                            'Metric {metric_time.val:.3f} ({metric_time.avg:.3f}) '
                            if args.time_breakdown else ''
                        ) +
                        'Remain {remain_time} '
                        'Loss total/contrastive/classifSketch/classifImage '
                        '{loss_meter.val:.4f}/{contrastive_loss_meter.val:.4f}/{sketch_loss_meter.val:.4f}/{image_loss_meter.val:.4f} '
                        'Top1classif sketch/image: {top1_s.val:.3f}/{top1_i.val:.3f}').format(epoch+1, sum_epochs, i + 1, len(train_loader),
                                                                        iter_time=iter_time,
                                                                        data_time=data_time,
                                                                        transfer_time=transfer_time,
                                                                        batch_time=batch_time,
                                                                        metric_time=metric_time,
                                                                        remain_time=remain_time,
                                                                        contrastive_loss_meter=loss_contras_meter,
                                                                        sketch_loss_meter=loss_sketch_meter,
                                                                        image_loss_meter=loss_image_meter,
                                                                        loss_meter=loss_meter,
                                                                        top1_s=top1_meter_sketch,
                                                                        top1_i=top1_meter_image))

    if main_process():
        logger.info('Train result at epoch [{}/{}]: Loss total/contras/classifSketch/classifImage {:.4f}/{:.4f}/{:.4f}/{:.4f}. '
                    'Top1classif sketch/image: {top1_s.avg:.3f}/{top1_i.avg:.3f}'.format(epoch+1, sum_epochs, loss_meter.avg, loss_contras_meter.avg, loss_sketch_meter.avg, loss_image_meter.avg, top1_s=top1_meter_sketch, top1_i=top1_meter_image))
    return loss_meter.avg

def triplet_train(train_loader, model, criterion, optimizer, epoch, classif_criterion):
    iter_time = AverageMeter()
    data_time = AverageMeter()
    transfer_time = AverageMeter()
    batch_time = AverageMeter()
    metric_time = AverageMeter()
    loss_meter = AverageMeter()
    loss_sketch_meter = AverageMeter()
    loss_positive_meter = AverageMeter()
    loss_negative_meter = AverageMeter()
    loss_triplet_meter = AverageMeter()
    top1_meter_sketch = AverageMeter()
    top1_meter_positive = AverageMeter()
    top1_meter_negative = AverageMeter()

    model.train()
    end = time.time()
    sum_epochs = args.contras_epochs + args.triplet_epochs
    max_iter = sum_epochs * len(train_loader)
    for i, (sketch_input, positive_input, negative_input, sketch_target, negative_target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if args.time_breakdown:
            last = time.time()

        sketch_input = sketch_input.cuda(non_blocking=True)
        positive_input = positive_input.cuda(non_blocking=True)
        negative_input = negative_input.cuda(non_blocking=True)
        sketch_target = sketch_target.cuda(non_blocking=True)
        negative_target = negative_target.cuda(non_blocking=True)
        if args.time_breakdown:
            torch.cuda.synchronize()
            transfer_time.update(time.time() - last)
            last = time.time()

        (sketchFeatVec, positiveFeatVec, negativeFeatVec), (sketch_output, positive_output, negative_output) = model(sketch_input, positive_input, negative_input)
        loss_sketch = smooth_loss(sketch_output, sketch_target, args.label_smoothing) if args.label_smoothing else classif_criterion(sketch_output, sketch_target)
        loss_positive = smooth_loss(positive_output, sketch_target, args.label_smoothing) if args.label_smoothing else classif_criterion(positive_output, sketch_target)
        loss_negative = smooth_loss(negative_output, negative_target, args.label_smoothing) if args.label_smoothing else classif_criterion(negative_output, negative_target)
        sketchFeatVec = F.normalize(sketchFeatVec)
        positiveFeatVec = F.normalize(positiveFeatVec)
        negativeFeatVec = F.normalize(negativeFeatVec)
        loss_triplet = criterion(sketchFeatVec, positiveFeatVec, negativeFeatVec)

        loss_triplet, loss_sketch, loss_positive, loss_negative = loss_triplet, loss_sketch*0.5, loss_positive*0.25, loss_negative*0.25
        classif_decay= 0.5*(1+ math.cos(epoch*math.pi/sum_epochs))
        loss = 2.0*loss_triplet*epoch + classif_decay*(loss_sketch + loss_positive + loss_negative)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.time_breakdown:
            torch.cuda.synchronize()
            batch_time.update(time.time() - last)
            last = time.time()

        n = sketch_input.size(0)
        if args.multiprocessing_distributed:
            with torch.no_grad():
                loss, loss_triplet, loss_sketch, loss_positive, loss_negative = loss.detach() * n, loss_triplet.detach() * n, loss_sketch.detach() *n, loss_positive.detach()*n, loss_negative.detach()*n
                count = sketch_target.new_tensor([n], dtype=torch.long)
                dist.all_reduce(loss), dist.all_reduce(count), dist.all_reduce(loss_triplet), dist.all_reduce(loss_sketch), dist.all_reduce(loss_positive), dist.all_reduce(loss_negative)
                n = count.item()
                loss, loss_triplet, loss_sketch, loss_positive, loss_negative = loss / n, loss_triplet/n, loss_sketch/n, loss_positive/n, loss_negative/n
        loss_meter.update(loss.item(), n)
        loss_triplet_meter.update(loss_triplet.item(), n)
        loss_sketch_meter.update(loss_sketch.item(), n)
        loss_positive_meter.update(loss_positive.item(), n)
        loss_negative_meter.update(loss_negative.item(), n)

        # classification metrics
        top1_s, = cal_accuracy(sketch_output, sketch_target, topk=(1,))
        top1_p, = cal_accuracy(positive_output, sketch_target, topk=(1,))
        top1_n, = cal_accuracy(negative_output, negative_target, topk=(1,))
        n = sketch_input.size(0)
        if args.multiprocessing_distributed:
            with torch.no_grad():
                top1_s = top1_s * n
                top1_p = top1_p * n
                top1_n = top1_n * n
                count = sketch_target.new_tensor([n], dtype=torch.long)
                dist.all_reduce(top1_s), dist.all_reduce(top1_p), dist.all_reduce(top1_n), dist.all_reduce(count)                
                n = count.item()
                top1_s = top1_s / n
                top1_p = top1_p / n    
                top1_n = top1_n / n 
        top1_meter_sketch.update(top1_s.item(), n), top1_meter_positive.update(top1_p.item(), n), top1_meter_negative.update(top1_n.item(), n)            

        if args.time_breakdown:
            torch.cuda.synchronize()
            metric_time.update(time.time() - last)
        iter_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * iter_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if ((i + 1) % args.print_freq == 0) and main_process():
            logger.info(('Epoch: [{}/{}][{}/{}] '
                        'Time {iter_time.val:.3f} ({iter_time.avg:.3f}) '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) ' +
                        (
                            'Transfer {transfer_time.val:.3f} ({transfer_time.avg:.3f}) '
                            'Batch {batch_time.val:.4f} ({batch_time.avg:.4f}) '
                            'Metric {metric_time.val:.3f} ({metric_time.avg:.3f}) '
                            if args.time_breakdown else ''
                        ) +
                        'Remain {remain_time} '
                        'Loss total/triplet/classifSketch/classifPos/classifNeg '
                        '{loss_meter.val:.4f}/{triplet_loss_meter.val:.4f}/{sketch_loss_meter.val:.4f}/{positive_loss_meter.val:.4f}/{negative_loss_meter.val:.4f} '
                        'Top1classif sketch/positive/negative {top1_s.val:.3f}/{top1_p.val:.3f}/{top1_n.val:.3f}').format(epoch+1, sum_epochs, i + 1, len(train_loader),
                                                                        iter_time=iter_time,
                                                                        data_time=data_time,
                                                                        transfer_time=transfer_time,
                                                                        batch_time=batch_time,
                                                                        metric_time=metric_time,
                                                                        remain_time=remain_time,
                                                                        triplet_loss_meter=loss_triplet_meter,
                                                                        sketch_loss_meter=loss_sketch_meter,
                                                                        positive_loss_meter=loss_positive_meter,
                                                                        negative_loss_meter=loss_negative_meter,
                                                                        loss_meter=loss_meter,
                                                                        top1_s=top1_meter_sketch,
                                                                        top1_p=top1_meter_positive,
                                                                        top1_n=top1_meter_negative))

    if main_process():
        logger.info('Train result at epoch [{}/{}]: Loss total/triplet/classifSketch/classifPos/classifNeg {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}. '
                    'Top1classif sketch/positive/negative {top1_s.avg:.3f}/{top1_p.avg:.3f}/{top1_n.avg:.3f}'.format(epoch+1, sum_epochs, loss_meter.avg, loss_triplet_meter.avg, loss_sketch_meter.avg, loss_positive_meter.avg, loss_negative_meter.avg, top1_s=top1_meter_sketch, top1_p=top1_meter_positive, top1_n=top1_meter_negative))
    return loss_meter.avg

@torch.no_grad()
def retrieval_validate(image_loader, sketch_loader, model, score_fun):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    iter_time = AverageMeter()
    data_time = AverageMeter()
    mAP = RetrievalMAP(compute_on_step=False)

    model.eval()
    end = time.time()

    # To process each image only once, we store query info in memory (it's enough with just queries,
    # and they are relatively few compared to catalog, so less memory required)
    processed_queries = []
    for (sketch_input, sketch_target, sketch_index) in sketch_loader:
        sketch_input = sketch_input.cuda(non_blocking=True)
        sketch_target = sketch_target.cuda(non_blocking=True)
        sketch_index = sketch_index.cuda(non_blocking=True)
        sketchFeatVec, _ = model(sketch_input, mode='sketch')
        processed_queries.append((sketchFeatVec, sketch_target, sketch_index))

    for i, (image_input, image_target) in enumerate(image_loader):
        data_time.update(time.time() - end)        
        image_input = image_input.cuda(non_blocking=True)
        image_target = image_target.cuda(non_blocking=True)
        imageFeatVec, _ = model(image_input, mode='image')

        for sketchFeatVec, sketch_target, sketch_index in processed_queries:
            scores = score_fun(imageFeatVec, sketchFeatVec)
            indices = torch.broadcast_to(sketch_index.unsqueeze(0), scores.size())
            target = image_target.unsqueeze(1) == sketch_target.unsqueeze(0)

            mAP(scores, target, indices)
            
        iter_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.test_print_freq == 0 and main_process():
            logger.info('Test batch: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Iter {iter_time.val:.3f} ({iter_time.avg:.3f})'.format(i + 1, len(image_loader),
                                                                        data_time=data_time,
                                                                        iter_time=iter_time))
    map_value = mAP.compute() # computes across all distributed processes
    if main_process():
        logger.info('Val result: mAP {:.4f}.'.format(map_value))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return map_value



if __name__ == '__main__':
    main()
