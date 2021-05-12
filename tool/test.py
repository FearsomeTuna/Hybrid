import os
import random
import time
import cv2
import numpy as np
import logging
import argparse
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import torch.distributed as dist

from model.san import san
from model.hybrid import hybrid
from model.resnet import resnet

from model.san import san
from util import config
from util.util import AverageMeter, intersectionAndUnionGPU, cal_accuracy

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/imagenet/imagenet_san10_pairwise.yaml', help='config file')
    parser.add_argument('opts', help='see config/imagenet/imagenet_san10_pairwise.yaml for all options', default=None, nargs=argparse.REMAINDER)
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
    return logger


def main():
    global args, logger
    args = get_parser()
    logger = get_logger()
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)

    n_channels = 3
    if args.channels: n_channels = args.channels

    if (args.arch == 'resnet'): # resnet
        model = resnet(args.layers, args.widths, args.classes, n_channels)
    elif args.arch == 'san': # SAN
        model = san(args.sa_type, args.layers, args.kernels, args.classes, in_planes=n_channels)
    elif args.arch == 'hybrid':
        model = hybrid(args.sa_type, args.layers, args.widths, args.kernels, args.classes, in_planes=n_channels)
    
    logger.info(model)
    model = torch.nn.DataParallel(model.cuda())
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    if os.path.isdir(args.save_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        logger.info("=> loaded checkpoint '{}'".format(args.model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if args.mean: mean = args.mean
    if args.std: std = args.std

    test_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)])
    if args.channels == 1:
        test_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), test_transform])

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)

    if args.split == 0:
        test_transform_set = torchvision.datasets.ImageFolder(os.path.join(args.data_root, 'data'), test_transform)
        split_ratio = args.split_ratio
        dataset_size = len(test_transform_set)
        indices = list(range(dataset_size))
        np.random.shuffle(indices)

        val_split = int(np.floor(split_ratio[0] * dataset_size)) # from index 0 to val_split - 1 is val set
        test_split = int(np.floor(split_ratio[1] * dataset_size)) + val_split # from index val_split to test_split is test set.

        if args.use_val_set:
            test_set = torch.utils.data.Subset(test_transform_set, indices[:val_split])
        else:
            test_set = torch.utils.data.Subset(test_transform_set, indices[val_split:test_split])
    elif args.split == 1 and args.use_val_set:
        test_transform_set = torchvision.datasets.ImageFolder(os.path.join(args.data_root, 'train'), test_transform)
        dataset_size = len(test_transform_set)
        indices = list(range(dataset_size))
        np.random.shuffle(indices)
        split = int(np.floor(split_ratio * dataset_size))

        val_indices = indices[:split]
        test_set = torch.utils.data.Subset(test_transform_set, val_indices)
    elif args.split == 1 and not args.use_val_set:
        test_set = torchvision.datasets.ImageFolder(os.path.join(args.data_root, 'test'), test_transform)
    elif args.split == 2:
        folder_name = 'val' if args.use_val_set else 'test'
        test_set = torchvision.datasets.ImageFolder(os.path.join(args.data_root, folder_name), test_transform)
    
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size_test, shuffle=False, num_workers=args.test_workers, pin_memory=True)
    validate(test_loader, model, criterion)


def validate(val_loader, model, criterion):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            output = model(input)
        loss = criterion(output, target)
        top1, top5 = cal_accuracy(output, target, topk=(1, 5))
        n = input.size(0)
        loss_meter.update(loss.item(), n), top1_meter.update(top1.item(), n), top5_meter.update(top5.item(), n)

        output = output.max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f} '
                        'Acc@1 {top1.val:.3f} ({top1.avg:.3f}) '
                        'Acc@5 {top5.val:.3f} ({top5.avg:.3f}).'.format(i + 1, len(val_loader),
                                                                        data_time=data_time,
                                                                        batch_time=batch_time,
                                                                        loss_meter=loss_meter,
                                                                        accuracy=accuracy,
                                                                        top1=top1_meter,
                                                                        top5=top5_meter))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('Val result: mIoU/mAcc/allAcc/top1/top5 {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc, top1_meter.avg, top5_meter.avg))
    for i in range(args.classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return loss_meter.avg, mIoU, mAcc, allAcc, top1_meter.avg, top5_meter.avg


if __name__ == '__main__':
    main()
