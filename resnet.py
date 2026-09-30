# -*- coding: utf-8 -*-
"""Standard ResNet encoder used by the published BEAM-Net configuration."""
import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, deep_base=False, stem_width=32):
        super().__init__()
        self.inplanes = stem_width * 2 if deep_base else 64
        if deep_base:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, stem_width, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stem_width), nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width, 3, padding=1, bias=False),
                nn.BatchNorm2d(stem_width), nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width * 2, 3, padding=1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        layers.extend(block(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x0 = x
        x = self.maxpool(x0)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x0, x1, x2, x3, x4]


def _build(block, layers, pretrained=False, **kwargs):
    arch = kwargs.pop('arch')
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls[arch]), strict=False)
    return model


def resnet18(pretrained=False, **kwargs):
    kwargs['arch'] = 'resnet18'
    return _build(BasicBlock, [2, 2, 2, 2], pretrained, **kwargs)


def resnet34(pretrained=False, **kwargs):
    kwargs['arch'] = 'resnet34'
    return _build(BasicBlock, [3, 4, 6, 3], pretrained, **kwargs)


def resnet50(pretrained=False, **kwargs):
    kwargs['arch'] = 'resnet50'
    return _build(Bottleneck, [3, 4, 6, 3], pretrained, **kwargs)


def resnet101(pretrained=False, **kwargs):
    kwargs['arch'] = 'resnet101'
    return _build(Bottleneck, [3, 4, 23, 3], pretrained, **kwargs)


def resnet152(pretrained=False, **kwargs):
    kwargs['arch'] = 'resnet152'
    return _build(Bottleneck, [3, 8, 36, 3], pretrained, **kwargs)
