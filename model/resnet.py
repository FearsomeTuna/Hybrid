from torch import Tensor
import torch.nn as nn
from typing import Callable, Optional

from torchvision.models.resnet import conv1x1, conv3x3
from util.util import init_weights

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, outplanes: int, stride: int = 1, downsample: Optional[nn.Module] = None,
                 groups: int = 1, base_width: int = 64, dilation: int = 1, norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        """Initializes a resnet bottleneck block with full preactivation. Based on torchvision postactivation implementation.

        This implementation places the stride for downsampling at 3x3 convolution(self.c
        while original implementation places the stride at the first 1x1 convolution(sel
        according to "Deep residual learning for image recognition"https://arxiv.org/abs
        This is based off of ResNet V1.5 (postactivation) and improves accuracy accordin
        https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

        Args:
            inplanes (int): input channels.
            outplanes (int): output channels. Must be divisible by self.expansion factor (4).
            stride (int): stride for middle convolution. This dictates spatial resolution downsampling.
            downsample (Optional[nn.Module]): module responsible for downsampling. Out dimensions must match with block output.
            groups (int): groups for middle convolution
            dilation (int): dilation for middle convolution
            norm_layer (Optional[Callable[..., nn.Module]]): batch normalization layer to be used. Defaults to torch.nn.BatchNorm2d
        """
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        assert outplanes % self.expansion == 0
        planes = outplanes // self.expansion
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.bn1 = norm_layer(inplanes)
        self.conv1 = conv1x1(inplanes, width)
        self.bn2 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn3 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        init_weights(self)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out