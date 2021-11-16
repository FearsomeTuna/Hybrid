import torch
import torch.nn as nn
from torch import Tensor
from functools import partial

from typing import Optional, List

from util.util import init_weights
from . import san
from . import resnet
from . import nl

import torchvision.models.resnet as torchres

#  some parts are adapted from torchvision.models.resnet ResNet._make_layer method implementation.

def _resnetStem():
    outplanes = 64
    conv1 = nn.Conv2d(3, outplanes, kernel_size=7, stride=2, padding=3,
                      bias=False)
    bn1 = nn.BatchNorm2d(outplanes)
    relu = nn.ReLU(inplace=True)
    maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    init_weights(conv1)
    init_weights(bn1)
    return nn.Sequential(conv1, bn1, relu, maxpool)

def _sanStem():
    outplanes = 64
    conv_in = nn.Conv2d(3, outplanes, kernel_size=1, bias=False)
    init_weights(conv_in)
    return conv_in

class MixedModel(nn.Module):
    def __init__(self, layers: List[int], layer_types: List[str], widths: List[int], num_classes: int, stem: str, sa_type: Optional[int] = None, added_nl_blocks: Optional[List[int]] = None):
        """Initializes model that can mix resnet, san and non-local based stages, and can
        add some non-local blocks within each stage type, following ResNetNLLayer and SanNLLayer design
        (non-local blocks can't be added on top if the respective stage is non-local based).

        SAN and non-local stages are preceded by transition layers that reduce spatial resolution to half, and can expand channels.
        Each stage reduces resolution to half, except when the first stage is resnet, in which case resolution is kept the same.
        San kernels are all 7x7, except in first stage, in which case kernel is 3x3 (following SAN original design).
        Batchnorm and ReLU are added after last stage if that last stage is san or nl based (following SAN original implementation).

        Both resnet and san stems output 64 channels. Additionally, ResNet stem reduces spatial resolution to 1/4 (224x224 to 56x56).
        San stem does not reduce spatial resolution.

        Model head is avgpool to 1x1, followed by fc layer with num_classes output size.

        Args:
            layers (List[int]): amount of blocks per each stage (not considering nl blocks).
            layers_type (List[str]): list indicating the type of block to use in every stage. Every element can be one of 'san', 'res' or 'nl'.
            widths (List[int]): list indicating output channel size for each stage.
            num_classes (int): amount of classes for classification
            stem (str): either 'res' (7x7 conv, stride 2 -> bn -> relu -> 2x2 maxpool) or 'san' (single 1x1 convolution). Both output 64 channels.
            sa_type (Optional[int]): san operation, 0 for pairwise, 1 for patchwise           
            added_nl_blocks (Optional[List[int]], optional): amount of non-local blocks to add per stage.
            Must be equal or less than number of blocks of respective stage. Defaults to None.
        """
        super().__init__()

        inplanes = 64
        nl_mode = 'dot'
        in_widths = [inplanes] + widths[:-1]
        assert len(layers) == len(layer_types)
        assert len(layers) == len(widths)

        if 'san' in layer_types:
            assert sa_type == 0 or sa_type == 1

        if added_nl_blocks == None:
            added_nl_blocks = [0]*len(layers)
        
        assert len(added_nl_blocks) == len(layers)
        
        for t, added_blocks in zip(layer_types, added_nl_blocks):
            if t == 'nl' and added_blocks > 0:
                raise ValueError('Bad argument: Non-Local stage cannot have added non-local blocks.')

        if stem == 'san':
            self.stem = _sanStem()
        elif stem == 'res':
            self.stem = _resnetStem()
        else:
            raise ValueError('Invalid stem argument.')
        backbone = []

        for i, stage_type, blocks, added_nl, _inplanes, _outplanes in zip(range(len(layers)), layer_types, layers, added_nl_blocks, in_widths, widths):
            layer = []
            if stage_type == 'san':
                layer.append(san.TransitionLayer(_inplanes, _outplanes))
                kernel_size = 3 if i == 0 else 7
                if added_nl > 0:
                    layer.append(SanNLLayer(blocks, _outplanes, added_nl, sa_type, nl_mode, kernel_size=kernel_size))
                else:
                    layer.append(SANLayer(sa_type, _outplanes, blocks, kernel_size=kernel_size))                
            elif stage_type == 'res':
                stride = 1 if i == 0 else 2 # first layer resnet type does not reduce resolution
                if added_nl > 0:
                    layer.append(ResNetNLLayer(blocks, _inplanes, _outplanes, added_nl, nl_mode, stride=stride))
                else:
                    layer.append(ResNetLayer(blocks, _inplanes, _outplanes, stride=stride))
            elif stage_type == 'nl':
                layer.append(san.TransitionLayer(_inplanes, _outplanes))
                layer.append(nl.NLLayer2D(blocks, _outplanes, mode=nl_mode))
            else:
                raise ValueError('stage_type argument includes "' + stage_type + '" invalid argument.')

            if i == len(layers)-1 and layer_types[-1] == 'san' or layer_types[-1] == 'nl': # last layer is san or nl
                layer.extend([nn.BatchNorm2d(widths[-1]), nn.ReLU(inplace=True)])
                init_weights(layer[-2])
            backbone.append(nn.Sequential(*layer))        

        self.backbone = nn.Sequential(*backbone)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[-1], num_classes)
    
    def forward(self, x: Tensor, getFeatVec: bool = False) -> Tensor:
        x = self.stem(x)
        x = self.backbone(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if getFeatVec:
            return x
        x = self.fc(x)
        return x


def _shortcut(inplanes: int, outplanes: int, stride: int) -> Optional[nn.Sequential]:
    shortcut = None
    if stride != 1 or inplanes != outplanes:
        shortcut = nn.Sequential(
            torchres.conv1x1(inplanes, outplanes, stride),
            nn.BatchNorm2d(outplanes),
        )
    init_weights(shortcut)
    return shortcut

class ResNetLayer(nn.Sequential):
    def __init__(self, blocks: int, inplanes: int, outplanes: int, stride: int = 1):
        """Initializes a sequence of blocks composing a typical resnet stage with bottleneck design.

        Args:
            blocks (int): number of resnet blocks
            inplanes (int): input planes
            outplanes (int): output planes
            stride (int, optional): stride for for first block. Defaults to 1.
        """
        shortcut = _shortcut(inplanes, outplanes, stride)

        layers = []
        layers.append(resnet.Bottleneck(inplanes, outplanes, stride, shortcut)) # first block possibly performs spatial reduction and channel expansion
    
        for _ in range(1, blocks):
            layers.append(resnet.Bottleneck(outplanes, outplanes))
        super().__init__(*layers)

class SANLayer(nn.Sequential):
    def __init__(self, sa_type: int, planes: int, blocks: int, kernel_size: int = 7, stride: int = 1) -> None:
        layers = []
        for _ in range(0, blocks):
            layers.append(san.Bottleneck(sa_type, planes, planes // 16, planes // 4, planes, 8, kernel_size, stride))
        super().__init__(*layers)

class ResNetNLLayer(nn.Sequential):
    def __init__(self, res_blocks: int, inplanes: int, outplanes: int, nl_blocks: int, nl_mode: str, stride: int = 1) -> None:
        """Initializes a sequence of resnet blocks with a number of non-local blocks added. If the amount of non-local blocks
        is half or less than the number of resnet blocks, then non-local blocks are added in between every other pair of resnet blocks,
        starting from the end. If res_blocks//2 > nl_blocks > res_blocks, non-local blocks are added between every pair of
        resnet blocks, starting from the end.

        Spatial dimension reduction depends on stride argument. Channel expansion depends on inplanes and outplanes arguments.
        If nl_blocks == res_blocks, then a small transition layer is added at the start, since non-local blocks don't
        handle spatial dimension reduction nor channel expansion.

        Example:
        6 res_blocks, 2 nl_blocks: [res, res, res, nl, res, res, nl, res]
        6 res_blocks, 3 nl_blocks: [res, nl, res, res, nl, res, res, nl, res]
        6 res_blocks, 4 nl_blocks: [res, res, nl, res, nl, res, nl, res, nl, res]
        1 res_blocks, 1 nl_blocks: [transition, nl, res]

        Args:
            res_blocks (int): Number of resnet blocks. Must be greater or equal to number of non-local blocks.
            inplanes (int): input channels
            outplanes (int): output channels
            nl_blocks (int): number of non-local blocks, minimum 1.
            nl_mode (str):  non-local operation. Either 'dot', 'embedded', 'concatenate' or 'gaussian'
            stride (int, optional): stride for for first block. Defaults to 1.
        """
        assert nl_blocks <= res_blocks
        assert nl_blocks > 0        

        resBlock = partial(resnet.Bottleneck, outplanes, outplanes)
        nlBlock = partial(nl.NLBlockND, in_channels=outplanes, mode=nl_mode, dimension=2)

        layers = []

        # since nl blocks don't reduce output spatial dimensions nor expand output channels, in case nl_blocks == res_blocks,
        # we add a transition layer at the start to handle that.
        if nl_blocks == res_blocks:
            layers.append(san.TransitionLayer(inplanes, outplanes))
            for _ in range(nl_blocks):
                layers.extend([nlBlock(), resBlock()])
            assert len(layers) == res_blocks + nl_blocks + 1
            super().__init__(*layers)
            return

        shortcut = _shortcut(inplanes, outplanes, stride)
        layers.append(resnet.Bottleneck(inplanes, outplanes, stride, shortcut)) # first block possibly performs spatial reduction and channel expansion

        if nl_blocks*2 <= res_blocks:
            layers.extend([resBlock() for _ in range(res_blocks - 2*nl_blocks)])
            for _ in range(nl_blocks-1):
                layers.extend([nlBlock(), resBlock(), resBlock()])
            layers.extend([nlBlock(), resBlock()])
        else:
            layers.extend([resBlock() for _ in range(res_blocks - nl_blocks - 1)])
            for _ in range(nl_blocks):
                layers.extend([nlBlock(), resBlock()])

        assert len(layers) == res_blocks + nl_blocks
        super().__init__(*layers)

class SanNLLayer(nn.Sequential):
    def __init__(self, san_blocks: int, inplanes: int, nl_blocks: int, sa_type: int, nl_mode: str, stride: int = 1, kernel_size: int = 7) -> None:
        """Initializes a sequence of san blocks with a number of non-local blocks added. If the amount of non-local blocks
        is half or less than the number of san blocks, then non-local blocks are added in between every other pair of san blocks,
        starting from the end. If san_blocks//2 > nl_blocks > san_blocks, non-local blocks are added between every pair of
        san blocks, starting from the end.
        Sequence output planes are the same amount as input planes.

        Example:
        6 san_blocks, 2 nl_blocks: [san, san, san, nl, san, san, nl, san]
        6 san_blocks, 3 nl_blocks: [san, nl, san, san, nl, san, san, nl, san]
        6 san_blocks, 4 nl_blocks: [san, san, nl, san, nl, san, nl, san, nl, san]

        Args:
            san_blocks (int): Number of san blocks. Must be greater or equal to the number of non-local blocks.
            inplanes (int): input channels
            nl_blocks (int): number of non-local blocks, minimum 1.
            sa_type (int): type of san block, 0 for pairwise, 1 for patchwise
            nl_mode (str): non-local operation. Either 'dot', 'embedded', 'concatenate' or 'gaussian'
            stride (int, optional): stride for SAN operation. Defaults to 1.
            kernel_size (int, optional): kernel size for SAN operation. Defaults to 7.
        """
        assert nl_blocks <= san_blocks
        assert nl_blocks > 0

        sanBlock = partial(san.Bottleneck, sa_type, inplanes, inplanes //
                           16, inplanes // 4, inplanes, 8, kernel_size, stride)
        nlBlock = partial(nl.NLBlockND, inplanes, mode=nl_mode, dimension=2)
        layers = []

        if nl_blocks*2 <= san_blocks:
            layers.extend([sanBlock()
                          for _ in range(san_blocks - 2*nl_blocks + 1)])
            for _ in range(nl_blocks-1):
                layers.extend([nlBlock(), sanBlock(), sanBlock()])
            layers.extend([nlBlock(), sanBlock()])
        else:
            layers.extend([sanBlock() for _ in range(san_blocks - nl_blocks)])
            for _ in range(nl_blocks):
                layers.extend([nlBlock(), sanBlock()])

        assert len(layers) == san_blocks + nl_blocks
        super().__init__(*layers)
