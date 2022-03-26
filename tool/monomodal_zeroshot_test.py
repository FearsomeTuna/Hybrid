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

from model.hybrid import MixedModel

from util import config
from util.util import AverageMeter, combination_cosine_similarity
from util.dataset import TxtFileDataset

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

    test_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(args.mean, args.std)])

    test_name = 'zeroshot_catalog'
    query_name = 'zeroshot_queries'
    if args.dataset_init == 'txt_mappings':
        test_set = TxtFileDataset(args.data_root, test_name+'.txt', 'mapping_zeroshot.txt', '\t', True, transform=test_transform)
        query_set = TxtFileDataset(args.data_root, query_name+'.txt', 'mapping_zeroshot.txt', '\t', True, transform=test_transform, return_index=True)
    else:
        raise ValueError("Invalid value for dataset_init config argument.")
    
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size_test, shuffle=False, num_workers=args.test_workers, pin_memory=True)
    query_loader = torch.utils.data.DataLoader(query_set, batch_size=args.batch_size_test, shuffle=False, num_workers=args.test_workers, pin_memory=True)

    validate(test_loader, query_loader, model, combination_cosine_similarity)

@torch.no_grad()
def validate(test_loader, query_loader, model, score_fun):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    iter_time = AverageMeter()
    data_time = AverageMeter()
    mAP = RetrievalMAP(compute_on_step=False)
    mRR = RetrievalMRR(compute_on_step=False)
    pAt10 = RetrievalPrecision(compute_on_step=False, k=10)

    model.eval()
    end = time.time()

    # To process each image only once, we store query info in memory (it's enough with just queries,
    # and they are relatively few compared to catalog, so less memory required)
    processed_queries = []
    for (query_input, query_target, query_index) in query_loader:
        query_input = query_input.cuda(non_blocking=True)
        query_target = query_target.cuda(non_blocking=True)
        query_index = query_index.cuda(non_blocking=True)
        query_embeding = model(query_input, getFeatVec=True)
        processed_queries.append((query_embeding, query_target, query_index))

    for i, (test_input, test_target) in enumerate(test_loader):
        data_time.update(time.time() - end)
        test_input = test_input.cuda(non_blocking=True)
        test_target = test_target.cuda(non_blocking=True)
        test_embeding = model(test_input, getFeatVec=True)

        for query_embeding, query_target, query_index in processed_queries:   
            scores = score_fun(test_embeding, query_embeding)
            indices = torch.broadcast_to(query_index.unsqueeze(0), scores.size())
            target = test_target.unsqueeze(1) == query_target.unsqueeze(0)
            
            mAP(scores, target, indices)
            mRR(scores, target, indices)
            pAt10(scores, target, indices)

        iter_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Iter {iter_time.val:.3f} ({iter_time.avg:.3f})'.format(i + 1, len(test_loader),
                                                                        data_time=data_time,
                                                                        iter_time=iter_time))
    map_value = mAP.compute()
    mrr_value = mRR.compute()
    pAt10_value = pAt10.compute()

    logger.info('Val result: mAP/mRR/P@10 {:.4f}/{:.4f}/{:.4f}.'.format(map_value, mrr_value, pAt10_value))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return map_value, mrr_value, pAt10_value


if __name__ == '__main__':
    main()
