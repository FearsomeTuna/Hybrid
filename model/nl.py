import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Type, Any, Callable, Union, List, Optional
from util.util import init_weights

from . import san

# Adapted from https://github.com/AlexHex7/Non-local_pytorch (Apache License 2.0) and https://github.com/tea1528/Non-Local-NN-Pytorch
class NLBlockND(nn.Module):
    def __init__(self, in_channels: int, inter_channels: Optional[int] = None, g_channels: Optional[int] = None, mode: str = 'embedded', 
                 dimension: int = 3, bn_layer: bool =True, sub_sample: bool =True) -> None:
        """Implementation of Non-Local Block with 4 different pairwise functions. Includes subsampling trick.
        Unlike original non-local block, we allow dedicated channel reduction for g function, instead of using the same
        for theta, phi and g. This is done to design and parameter count closer to SAN block for comparison.
        1x1 convolutions are initialized with kaiming normal.

        args:
            in_channels: input channel size
            inter_channels: channel size inside the block for theta and phi. Reduced to 1/8 of the input channel size if not specified
            g_channel: channel size inside the block for g function. Reduced to 1/4 of the input channel size if not specified
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
            sub_sample: whether to use sub sampling trick described in paper.
        """
        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]
        
        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')
            
        self.mode = mode
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.g_channels = g_channels

        # the channel size is reduced inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 8
            if self.inter_channels == 0:
                self.inter_channels = 1

        if self.g_channels is None:
            self.g_channels = in_channels // 4
            if self.g_channels == 0:
                self.g_channels = 1
        
        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.g_channels, kernel_size=1)
        init_weights(self.g)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                    conv_nd(in_channels=self.g_channels, out_channels=self.in_channels, kernel_size=1),
                    bn(self.in_channels)
                )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            init_weights(self.theta)
            init_weights(self.phi)

            if self.sub_sample:
                self.g = nn.Sequential(self.g, max_pool_layer)
                self.phi = nn.Sequential(self.phi, max_pool_layer)
        
        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                    nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                    nn.ReLU()
                )
            init_weights(self.W_f)
            
    def forward(self, x: Tensor) -> Tensor:
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        batch_size = x.size(0)
        
        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x).view(batch_size, self.g_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)
            
            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)
            
            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))
        
        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1) # number of position in x
            f_div_C = f / N
        
        y = torch.matmul(f_div_C, g_x)
        
        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.g_channels, *x.size()[2:])
        
        W_y = self.W_z(y)
        # residual connection
        z = W_y + x

        return z

def make_layer2D(blocks: int, inplanes: int, mode: str) -> nn.Sequential:
    layer = []
    for _ in range (blocks):
        layer.append(nn.BatchNorm2d(inplanes))
        layer.append(nn.ReLU(inplace=True))
        layer.append(NLBlockND(inplanes, mode=mode, dimension=2))
    init_weights(layer[0])
    return nn.Sequential(*layer)

class PureNonLocal2D(nn.Module):
    def __init__(self, layers: List[int], num_classes: int, grayscale: bool, mode: str) -> None:
        # doesn't rely on convolutions other than 1x1, except for stem
        super().__init__()

        assert len(layers) == 4
        self.layers = layers
        self.num_classes = num_classes
        self.grayscale = grayscale
        self.inplanes = 64
        self.mode = mode

        # stem
        self.stem = nn.Conv2d(1 if self.grayscale else 3, self.inplanes, kernel_size=7, stride=2, padding=3,
                              bias=False)
        init_weights(self.stem)

        self.transition1 = san.TransitionLayer(self.inplanes, 256)
        self.layer1 = make_layer2D(layers[0], 256, mode=self.mode)
        self.transition2 = san.TransitionLayer(256, 512)
        self.layer2 = make_layer2D(layers[1], 512, mode=self.mode)
        self.transition3 = san.TransitionLayer(512, 1024)
        self.layer3 = make_layer2D(layers[2], 1024, mode=self.mode)
        self.transition4 = san.TransitionLayer(1024, 2048)
        self.layer4 = make_layer2D(layers[3], 2048, mode=self.mode)

        self.bn = nn.BatchNorm2d(2048)
        init_weights(self.bn)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)

        x = self.transition1(x)
        x = self.layer1(x)
        x = self.transition2(x)
        x = self.layer2(x)
        x = self.transition3(x)
        x = self.layer3(x)
        x = self.transition4(x)
        x = self.layer4(x)

        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
