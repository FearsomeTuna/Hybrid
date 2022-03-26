import torch

from model.hybrid import MixedModel
from util.complexity import get_model_complexity_info
from util import config

with torch.cuda.device(0):
    print_model = False

    config_files = [
                     "config/sketch_eitz/sketch_eitz_resnet26.yaml",
                     "config/sketch_eitz/sketch_eitz_resnet26_nl.yaml",
                     "config/sketch_eitz/sketch_eitz_san10_pairwise.yaml",
                     "config/sketch_eitz/sketch_eitz_san10_pairwise_nl.yaml",
                     "config/sketch_eitz/sketch_eitz_san10_patchwise.yaml",
                     "config/sketch_eitz/sketch_eitz_san10_patchwise_nl.yaml",
                     "config/sketch_eitz/sketch_eitz_hybrid_patch_2-2.yaml",
                     "config/sketch_eitz/sketch_eitz_hybrid_patch_3-1.yaml",
                     "config/sketch_eitz/sketch_eitz_hybrid_pair_2-2.yaml",
                     "config/sketch_eitz/sketch_eitz_hybrid_pair_3-1.yaml",
                     "config/sketch_eitz/sketch_eitz_hybrid_patch_2-2_nl.yaml",
                     "config/sketch_eitz/sketch_eitz_hybrid_patch_3-1_nl.yaml",
                     "config/sketch_eitz/sketch_eitz_hybrid_pair_2-2_nl.yaml",
                     "config/sketch_eitz/sketch_eitz_hybrid_pair_3-1_nl.yaml",
                     "config/sketch_eitz/sketch_eitz_nl.yaml",
                     "config/sketch_eitz/sketch_eitz_hybrid_nl_2-2.yaml",
                     "config/sketch_eitz/sketch_eitz_hybrid_nl_3-1.yaml",
                     
    ]
    for config_file in config_files:
      args = config.load_cfg_from_cfg_file(config_file)
      model = MixedModel(args.layers, args.layer_types, args.widths, args.classes, args.layer_types[0], args.sa_type if 'san' in args.layer_types else None, args.added_nl_blocks)

      flops, params = get_model_complexity_info(model.cuda(), (3, 224, 224), as_strings=True, print_per_layer_stat=print_model)
      print('ConfigFile: {}.\nParams/Flops: {}/{}\n'.format(config_file, params, flops))