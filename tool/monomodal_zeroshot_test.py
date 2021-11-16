import os
import random
import time
import cv2
import numpy as np
import logging
import argparse
import shutil

from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import torch.distributed as dist


from model.san import san
from model.hybrid import MixedModel
from model.resnet import resnet
from model.nl import PureNonLocal2D

from util import config
from util.util import AverageMeter, combination_cosine_distance
from util.dataset import PathsFileDataset

from torchmetrics import RetrievalMAP, RetrievalMRR, RetrievalPrecision

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

    if (args.arch == 'resnet'): # resnet
        model = resnet(args.layers, args.classes)
    elif args.arch == 'san': # SAN
        model = san(args.sa_type, args.layers, args.kernels, args.classes)
    elif args.arch == 'nl':
        model = PureNonLocal2D(args.layers, args.classes, 'dot')
    elif args.arch == 'hybrid':
        model = MixedModel(args.layers, args.layer_types, args.widths, args.classes, args.layer_types[0], args.sa_type if 'san' in args.layer_types else None, args.added_nl_blocks)
    
    logger.info(model)
    model = torch.nn.DataParallel(model.cuda())

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

    test_name = 'zeroshot_test'
    query_name = 'zeroshot_queries'
    if args.dataset_init == 'preprocessed_paths':
        test_set = PathsFileDataset(args.data_root, test_name + '_init.pt', transform=test_transform)
        query_set = PathsFileDataset(args.data_root, query_name + '_init.pt', transform=test_transform)
    elif args.dataset_init == 'image_folder':
        test_set = torchvision.datasets.ImageFolder(os.path.join(args.data_root, test_name), test_transform)
        query_set = torchvision.datasets.ImageFolder(os.path.join(args.data_root, query_name), test_transform)
    else:
        raise ValueError("Invalid value for dataset_init config argument.")
    
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size_test, shuffle=False, num_workers=args.test_workers, pin_memory=True)
    query_loader = torch.utils.data.DataLoader(query_set, batch_size=args.batch_size_test, shuffle=False, num_workers=args.test_workers, pin_memory=True)

    logger.info('Using cosine similarity as score function.')
    validate(test_loader, query_loader, model, combination_cosine_distance)

def validate(test_loader, query_loader, model, score_fun):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    mAP = RetrievalMAP(compute_on_step=False)
    mRR = RetrievalMRR(compute_on_step=False)
    pAt10 = RetrievalPrecision(compute_on_step=False, k=10)

    model.eval()
    end = time.time()
    
    for i, (test_input, test_target) in enumerate(test_loader):
        data_time.update(time.time() - end)
        test_input = test_input.cuda(non_blocking=True)
        test_target = test_target.cuda(non_blocking=True)
        with torch.no_grad():
            test_embeding = model(test_input, getFeatVec=True)
        start_query_index = 0 
        for j, (query_input, query_target) in enumerate(query_loader):
            query_input = query_input.cuda(non_blocking=True)
            query_target = query_target.cuda(non_blocking=True)                     
            with torch.no_grad():
                query_embeding = model(query_input, getFeatVec=True)                
            scores = score_fun(test_embeding, query_embeding)

            query_batchsize = list(query_embeding.size())[0]
            end_query_index = start_query_index + query_batchsize
            indices = torch.arange(start_query_index, end_query_index)
            start_query_index = end_query_index
            indices = torch.broadcast_to(indices.unsqueeze(0), scores.size())

            target = test_target.unsqueeze(1) == query_target.unsqueeze(0)            
            mAP(scores, target, indices)
            mRR(scores, target, indices)
            pAt10(scores, target, indices)

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(i + 1, len(test_loader),
                                                                        data_time=data_time,
                                                                        batch_time=batch_time))
    map_value = mAP.compute()
    mrr_value = mRR.compute()
    pAt10_value = pAt10.compute()

    logger.info('Val result: mAP/mRR/P@10 {:.4f}/{:.4f}/{:.4f}.'.format(map_value, mrr_value, pAt10_value))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return map_value, mrr_value, pAt10_value


if __name__ == '__main__':
    main()
