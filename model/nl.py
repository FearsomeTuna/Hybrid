import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Type, Any, Callable, Union, List, Optional
from util.util import init_weights

from . import san

# Adapted from https://github.com/AlexHex7/Non-local_pytorch (Apache License 2.0) and https://github.com/tea1528/Non-Local-NN-Pytorch
class NLBlock(nn.Module):
    def __init__(self, in_channels: int, inter_channels: Optional[int] = None, g_channels: Optional[int] = None, mode: str = 'dot',
                 sub_sample: bool =True) -> None:
        """Implementation of Non-Local Block. Includes subsampling trick.
        Unlike original non-local block, we use a bottleneck design similar to that of SANet, and allow separate channel reduction for g function,
        instead of using the same for theta, phi and g. Also, block follows a preactivation scheme, rather than a post activation one.
        In total, one batchnorm layer and two ReLU activations are added (over the version with batchnorm), which were not in original design.
        These changes are intended to make design and parameter count closer to SAN block for comparison.
        1x1 convolutions are initialized with kaiming normal.

        args:
            in_channels: input channel size
            inter_channels: channel size inside the block for theta and phi. Reduced to 1/8 of the input channel size if not specified
            g_channel: channel size inside the block for g function. Reduced to 1/4 of the input channel size if not specified
            mode: currently supports only dot product
            sub_sample: whether to use sub sampling trick described in paper.
        """
        super(NLBlock, self).__init__()

        if mode != 'dot':
            raise NotImplementedError()
        self.mode = mode
        
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.g_channels = g_channels

        # by default, channel size inside block is reduced by factor of 8 for theta and phi
        if self.inter_channels is None:
            self.inter_channels = in_channels // 8
            if self.inter_channels == 0:
                self.inter_channels = 1
        # by default, channel size inside block is reduced by factor of 4 for g
        if self.g_channels is None:
            self.g_channels = in_channels // 4
            if self.g_channels == 0:
                self.g_channels = 1        

        self.bn = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.g_channels, kernel_size=1)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        self.W_z = nn.Sequential(
                nn.BatchNorm2d(self.g_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=self.g_channels, out_channels=self.in_channels, kernel_size=1)
                
            )    

        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        if self.sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

        # since we're doing preactivation, initialization of last batchnorm weights can no longer define
        # initial state of non-local block as identity mapping, like in section 4.1 of the paper.
        # we opt for initializing the whole block the same way we do resnet and san blocks
        init_weights(self)
            
    def forward(self, x: Tensor) -> Tensor:
        """
        args
            x: (N, C, H, W) for dimension 2
        """

        identity = x
        batch_size = x.size(0)
        
        x = self.relu(self.bn(x))

        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x).view(batch_size, self.g_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        f = torch.matmul(theta_x, phi_x)
        
        N = f.size(-1) # number of position in x
        f_div_C = f / N
        
        y = torch.matmul(f_div_C, g_x)
        
        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.g_channels, *x.size()[2:])
        
        W_y = self.W_z(y)
        # residual connection
        z = W_y + identity

        return z

class NLLayer(nn.Sequential):
    def __init__(self, blocks: int, inplanes: int, mode: str) -> None:
        if mode != 'dot':
            raise NotImplementedError()

        layer = []
        for _ in range (blocks):
            layer.append(NLBlock(inplanes, mode=mode))
        super().__init__(*layer)
