import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional

import torchvision.models.resnet as torchres
from util.util import init_weights

class ResNet(torchres.ResNet):
    def __init__(self, layers: List[int], num_classes: int = 1000) -> None:
        super().__init__(torchres.Bottleneck, layers, num_classes=num_classes)
    
    def forward(self, x: Tensor, getFeatVec: bool=False) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if getFeatVec:
            return x
        x = self.fc(x)
        return x

class Bottleneck(torchres.Bottleneck):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    
    def __init__(self, inplanes: int, outplanes: int, stride: int = 1, downsample: Optional[nn.Module] = None, groups: int = 1, base_width: int = 64, dilation: int = 1, norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        # in torchvision implementation, Bottleneck 'planes' argument corresponds to output planes but is internally expanded by res.Bottleneck.expansion factor (equal to 4).
        # Hence unintuitive instantiations like res.Bottleneck(256, 64), which is a block with 256 input planes and 256 output planes
        # We abstract this for clarity when declaring this adapted version.
        assert outplanes % torchres.Bottleneck.expansion == 0
        planes = outplanes // torchres.Bottleneck.expansion
        super().__init__(inplanes, planes, stride=stride, downsample=downsample, groups=groups, base_width=base_width, dilation=dilation, norm_layer=norm_layer)
        init_weights(self)

def resnet(layers, num_classes):
    model = ResNet(layers=layers, num_classes=num_classes)
    return model