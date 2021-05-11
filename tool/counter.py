import torch

from model.san import san
from model.hybrid import hybrid
from model.resnet import resnet
from util.complexity import get_model_complexity_info
from util import config

with torch.cuda.device(0):
    print_model = False

    config_files = [
                     "config/sketch_eitz/sketch_eitz_resnet26.yaml",
                     "config/sketch_eitz/sketch_eitz_san10_pairwise.yaml",
                     "config/sketch_eitz/sketch_eitz_san10_patchwise.yaml",
                     "config/sketch_eitz/sketch_eitz_hybrid_patch_2-2.yaml",
                     "config/sketch_eitz/sketch_eitz_hybrid_patch_3-1.yaml"
    ]
    for config_file in config_files:
      args = config.load_cfg_from_cfg_file(config_file)

      n_channels = 3
      if args.channels: n_channels = args.channels

      if args.arch == 'resnet':
          model = resnet(args.layers, args.widths, args.classes, n_channels)
      elif args.arch == 'san':
          model = san(args.sa_type, args.layers, args.kernels, args.classes, in_planes=n_channels)
      elif args.arch == 'hybrid':
          model = hybrid(args.sa_type, args.layers, args.widths, args.kernels, args.classes, in_planes=n_channels)

      flops, params = get_model_complexity_info(model.cuda(), (n_channels, 224, 224), as_strings=True, print_per_layer_stat=print_model)
      print('ConfigFile: {}.\nParams/Flops: {}/{}\n'.format(config_file, params, flops))