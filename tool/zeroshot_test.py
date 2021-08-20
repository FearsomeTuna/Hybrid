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
from model.hybrid import hybrid
from model.resnet import resnet

from util import config
from util.util import AverageMeter, intersectionAndUnionGPU, cal_accuracy
from util.dataset import PathsFileDataset, InMemoryDataset, pil_loader

from torchmetrics import RetrievalMAP, RetrievalMRR

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

def get_activation(mode):
    def hook(model, input, output):
        if mode == 'input':
            value = input[0]
        elif mode == 'output':
            value = output
        global activation
        activation = value.detach()
    return hook

def combination_cosine_distance(mat1, mat2):
    """Calculates cosine similarity between every vector of mat1 and every vector of mat2.

    If vectors are of length f, then input dimensions must be (N, f) and (M, f), where N and M > 0, and result will be a tensor of dimensions (N, M).
    If result = bipartite_cosine_distance(mat1, mat2), then result[i,j] is roughly equivalent to the value held by tensor
    torch.nn.functional.cosine_similarity(mat1[i].unsqueeze(0), mat2[j].unsqueeze(0)) for every position i, j.

    Args:
        mat1 (torch.tensor): first input
        mat2 (torch.tensor): second input

    Returns:
        torch.tensor: tensor with cosine distance results for every pairwise combination of vectors from mat1 to mat2.
    """
    mat1 = F.normalize(mat1)
    mat2 = F.normalize(mat2)
    return torch.mm(mat1, torch.transpose(mat2, 0, 1))

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

    n_channels = 3
    if args.channels: n_channels = args.channels

    if (args.arch == 'resnet'): # resnet
        model = resnet(args.layers, args.widths, args.classes, n_channels)
        model.head.flat.register_forward_hook(get_activation('output'))
    elif args.arch == 'san': # SAN
        model = san(args.sa_type, args.layers, args.kernels, args.classes, in_planes=n_channels)
        model.fc.register_forward_hook(get_activation('input'))
    elif args.arch == 'hybrid':
        model = hybrid(args.sa_type, args.layers, args.widths, args.kernels, args.classes, in_planes=n_channels)
        model.fc.register_forward_hook(get_activation('input'))
    
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

    if args.dataset_init == 'byte_imgs':
        test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        test_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)])

    test_name = 'zeroshot_test'
    query_name = 'zeroshot_queries'
    if args.dataset_init == 'preprocessed_paths':
        test_set = PathsFileDataset(args.data_root, test_name + '_init.pt', transform=test_transform)
        query_set = PathsFileDataset(args.data_root, query_name + '_init.pt', transform=test_transform)
    elif args.dataset_init == 'image_folder':
        test_set = torchvision.datasets.ImageFolder(os.path.join(args.data_root, test_name), test_transform, loader=pil_loader)
        query_set = torchvision.datasets.ImageFolder(os.path.join(args.data_root, query_name), test_transform, loader=pil_loader)
    elif args.dataset_init == 'byte_imgs':
        test_set = InMemoryDataset(os.path.join(args.data_root, test_name + '_imgs.pt'), test_transform)
        query_set = InMemoryDataset(os.path.join(args.data_root, query_name + '_imgs.pt'), test_transform)
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

    model.eval()
    end = time.time()
    
    for i, (test_input, test_target) in enumerate(test_loader):
        data_time.update(time.time() - end)
        test_input = test_input.cuda(non_blocking=True)
        test_target = test_target.cuda(non_blocking=True)
        start_query_index = 0 
        for j, (query_input, query_target) in enumerate(query_loader):
            query_input = query_input.cuda(non_blocking=True)
            query_target = query_target.cuda(non_blocking=True)                     
            with torch.no_grad():
                model(query_input)
                query_embeding = activation
                model(test_input)
                test_embeding = activation
            scores = score_fun(test_embeding, query_embeding)

            query_batchsize = list(query_embeding.size())[0]
            end_query_index = start_query_index + query_batchsize
            indices = torch.arange(start_query_index, end_query_index)
            start_query_index = end_query_index
            indices = torch.broadcast_to(indices.unsqueeze(0), scores.size())

            target = test_target.unsqueeze(1) == query_target.unsqueeze(0)            
            mAP(scores, target, indices)
            mRR(scores, target, indices)

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

    logger.info('Val result: mAP/mRR {:.4f}/{:.4f}.'.format(map_value, mrr_value))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return map_value, mrr_value


if __name__ == '__main__':
    main()
