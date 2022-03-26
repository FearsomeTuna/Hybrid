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
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F

from model.hybrid import MixedBiModal

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

    model = MixedBiModal(args.layers, args.layer_types, args.widths, args.classes, args.layer_types[0], args.share_layers, args.sa_type if 'san' in args.layer_types else None, args.added_nl_blocks)

    logger.info(model)
    model = torch.nn.DataParallel(model.cuda())

    if os.path.isdir(args.save_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        logger.info("=> loaded checkpoint '{}'".format(args.model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

    image_mean, image_std = args.image_mean, args.image_std
    sketch_mean, sketch_std = args.sketch_mean, args.sketch_std

    image_test_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(image_mean, image_std)])
    sketch_test_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(sketch_mean, sketch_std)])

    sketch_set = TxtFileDataset(args.test_sketch_root, 'test.txt', 'mapping.txt', '\t', True, transform=sketch_test_transform, return_index=True)
    image_set = TxtFileDataset(args.test_image_root, 'test.txt', 'mapping.txt', '\t', True, transform=image_test_transform)
    
    image_loader = torch.utils.data.DataLoader(image_set, batch_size=args.batch_size_test, shuffle=False, num_workers=args.test_workers, pin_memory=True)
    sketch_loader = torch.utils.data.DataLoader(sketch_set, batch_size=args.batch_size_test, shuffle=False, num_workers=args.test_workers, pin_memory=True)

    validate(image_loader, sketch_loader, model, combination_cosine_similarity)

@torch.no_grad()
def validate(image_loader, sketch_loader, model, score_fun):
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
        imageFeatVec, _ =  model(image_input, mode='image')

        for sketchFeatVec, sketch_target, sketch_index in processed_queries:
            scores = score_fun(F.normalize(imageFeatVec), F.normalize(sketchFeatVec))
            indices = torch.broadcast_to(sketch_index.unsqueeze(0), scores.size())            
            target = image_target.unsqueeze(1) == sketch_target.unsqueeze(0)

            mAP(scores, target, indices)
            mRR(scores, target, indices)
            pAt10(scores, target, indices)
        iter_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.test_print_freq == 0:
            logger.info('Test batch: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {iter_time.val:.3f} ({iter_time.avg:.3f})'.format(i + 1, len(image_loader),
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
