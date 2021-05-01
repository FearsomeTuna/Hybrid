from glasses.models import ResNet
from glasses.models.classification.resnet import ResNetBottleneckBlock

def resnet(layers, widths, num_classes, in_planes):
    model = ResNet(in_channels=in_planes, n_classes=num_classes, block=ResNetBottleneckBlock, widths=widths, depths=layers)
    return model