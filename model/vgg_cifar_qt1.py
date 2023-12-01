import time
import math
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from model.modules import *


defaultcfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 512]
# relu layer config
relucfg = [2, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39, 42]
# convolution layer config
convcfg = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]


# VGG
class QVGG_A(nn.Module):
    def __init__(self, num_classes, init_weights, cfg, filters_left,ka,kw):
        super(QVGG_A, self).__init__()
        self.features = nn.Sequential()
        if cfg is None:
            cfg = [filters_left[0], filters_left[1], 'M', filters_left[2], filters_left[3], 'M', filters_left[4], filters_left[5], filters_left[6], 'M', filters_left[7], filters_left[8], filters_left[9], 'M', filters_left[10], filters_left[11], filters_left[12], 512]
        self.relucfg = relucfg
        self.covcfg = convcfg
        self.features = self.make_layers(cfg[:-1], True, ka)
        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(cfg[-2], cfg[-1])),
            ('norm1', nn.BatchNorm1d(cfg[-1])),
            ('relu1', activation_quantize_fn(ka[-1])),
            ('linear2', nn.Linear(cfg[-1], num_classes)),
        ]))

        # Initialize weights
        if init_weights:
            self._initialize_weights()


    def make_layers(self, cfg, batch_norm, ka):
        layers = nn.Sequential()
        in_channels = 3
        cnt = 0
        for i, v in enumerate(cfg):
            if v == 'M':
                layers.add_module('pool%d' % i, nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                cnt += 1
                layers.add_module('conv%d' % i, conv2d)
                layers.add_module('norm%d' % i, nn.BatchNorm2d(v))
                layers.add_module('relu%d' % i, activation_quantize_fn(ka[i]))
                in_channels = v
        return layers


    def forward(self, x):
        x = self.features(x)

        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    # Initialize Weights Function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n)) 
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def qvgg_16_bn_A(filters_left, bit):
    return QVGG_A(num_classes=10, init_weights=True, cfg=None, filters_left=filters_left,ka=bit, kw=bit)




