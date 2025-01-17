import os
import time
import cv2
import numpy as np
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

from model.hybrid import MixedModel

from util import config
from util.util import AverageMeter, intersectionAndUnionGPU, cal_accuracy
from util.dataset import TxtFileDataset

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
    logger.propagate = False
    return logger


def main():
    global args, logger
    args = get_parser()
    logger = get_logger()
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)

    model = MixedModel(args.layers, args.layer_types, args.widths, args.classes, args.layer_types[0], args.sa_type if 'san' in args.layer_types else None, args.added_nl_blocks)
    
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

    folder_name = 'val' if args.use_val_set else 'test'
    if args.dataset_init == 'image_folder':
        test_set = torchvision.datasets.ImageFolder(os.path.join(args.data_root, folder_name), test_transform)
    elif args.dataset_init == 'txt_mappings':
        test_set = TxtFileDataset(args.data_root, folder_name + '.txt', 'mapping.txt', "\t", True, transform=test_transform)
    else:
        raise ValueError("Invalid value for dataset_init config argument.")
    
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size_test, shuffle=False, num_workers=args.test_workers, pin_memory=True)
    validate(test_loader, model, criterion)

@torch.no_grad()
def validate(val_loader, model, criterion):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    iter_time = AverageMeter()
    data_time = AverageMeter()
    transfer_time = AverageMeter() 
    batch_time = AverageMeter()
    metric_time = AverageMeter()
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
        if args.time_breakdown:
            last = time.time()

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        if args.time_breakdown:
            torch.cuda.synchronize()
            transfer_time.update(time.time() - last)
            last = time.time()

        output = model(input)
        loss = criterion(output, target)
        if args.time_breakdown:
            torch.cuda.synchronize()
            batch_time.update(time.time() - last)
            last = time.time()

        top1, top5 = cal_accuracy(output, target, topk=(1, 5))
        n = input.size(0)
        loss_meter.update(loss.item(), n), top1_meter.update(top1.item(), n), top5_meter.update(top5.item(), n)

        output = output.max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        if args.time_breakdown:
            torch.cuda.synchronize()
            metric_time.update(time.time() - last)
        iter_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            logger.info(('Test: [{}/{}] '
                        'Time {iter_time.val:.3f} ({iter_time.avg:.3f}) '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) ' +
                        (
                            'Transfer {transfer_time.val:.3f} ({transfer_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Metric {metric_time.val:.3f} ({metric_time.avg:.3f}) '
                            if args.time_breakdown else ''
                        ) +   
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f} '
                        'Acc@1 {top1.val:.3f} ({top1.avg:.3f}) '
                        'Acc@5 {top5.val:.3f} ({top5.avg:.3f}).').format(i + 1, len(val_loader),
                                                                        iter_time=iter_time,
                                                                        data_time=data_time,
                                                                        transfer_time=transfer_time,
                                                                        batch_time=batch_time,
                                                                        metric_time=metric_time,
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
