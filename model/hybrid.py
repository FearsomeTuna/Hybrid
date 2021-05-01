import torch
import torch.nn as nn

from . import san

import glasses.models.classification.resnet as resnet


class Hybrid(nn.Module):
    """
    Hybrid classification model whose encoder starts with resnet architecture and ends with san architecture.
    """
    def __init__(self, sa_type, block, layers, widths_resnet, kernels_san, num_classes, in_planes=3):
        super(Hybrid, self).__init__()

        resnet_stages = len(widths_resnet)
        san_stages = len(kernels_san)
        assert len(layers) == resnet_stages + san_stages

        # resnet is the first part of the model
        layers_resnet = layers[:resnet_stages]
        layers_san = layers[resnet_stages:]

        self.resnet_encoder = resnet.ResNetEncoder(in_channels=in_planes, start_features=64, widths=widths_resnet, depths = layers_resnet, block=resnet.ResNetBottleneckBlock)

        # variable number of san stages and transitions
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.layer = nn.ModuleList()

        c = widths_resnet[-1]

        for depth, kernel_size in zip(layers_san, kernels_san):
            c *= 2
            self.conv.append(san.conv1x1(c // 2, c))
            self.bn.append(nn.BatchNorm2d(c))
            self.layer.append(san.make_layer(sa_type, block, c, depth, kernel_size))

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c, num_classes)


    def forward(self, x):
        x = self.resnet_encoder(x)

        for i in range(len(self.conv)):
            x = self.relu(self.bn[i](self.layer[i](self.conv[i](self.pool(x)))))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def hybrid(sa_type, layers, widths_resnet, kernels_san, num_classes, in_planes=3):
    model = Hybrid(sa_type, san.Bottleneck, layers, widths_resnet, kernels_san, num_classes, in_planes)
    return model
