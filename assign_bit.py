import torch
import numpy as np
import torch.nn as nn
import math
from model.resnet_cifar import *
from model.MobileNetV2 import InvertedResidual

def get_num_gen(gen):
    return sum(1 for x in gen)
def is_leaf(model):
    return get_num_gen(model.children()) == 0

class BitwidthAllocatorMBNetV2(object):
    def __init__(self, model, device='cuda'):
        self.model = model
        self.magnitude_statistics = {}
        self.device = device
    def forward(self, x):
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model
        
        activation_index = 0
        for l1, m1 in enumerate(model.features.children()):
            skipped = False
            if isinstance(m1, InvertedResidual):
                if m1.use_res_connect:
                    skipped = True
                # m1 is nn.Sequential now
                m1 = m1.conv 

            # use for residual
            tmp_x = x 

            cnt = 0
            for l2, m2 in enumerate(m1.children()):
                cnt += 1
                x = m2(x)
                if isinstance(m2, nn.Conv2d):
                    values = (torch.abs(m2.weight.data)).sum(1).sum(1).sum(1)
                    self.magnitude_statistics[activation_index] = values
                    activation_index += 1
            if cnt == 0:
                x = m1(x)

        return model.classifier(x.view(x.size(0), -1)), self.magnitude_statistics

    
class BitwidthAllocatorVGG(object):
    def __init__(self, model, device='cuda'):
        self.model = model
        self.magnitude_statistics = {}
        self.device = device
    def forward(self, x):
        if isinstance(self.model, nn.DataParallel):
            model = self.model.module
        else:
            model = self.model
        
        activation_index = 0
        for layer, method in enumerate(model.features.children()):
            if isinstance(method, nn.Conv2d):
                values = (torch.abs(method.weight.data)).sum(1).sum(1).sum(1).sum(0)
                self.magnitude_statistics[activation_index] = values
                activation_index += 1
            x = method(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        return x, self.magnitude_statistics

class BitwidthAllocatorResNet(object):
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

    def trace_layer(self, layer, x):
        y = layer.old_forward(x)
        if isinstance(layer, nn.Conv2d):
            values = (torch.abs(layer.weight.data)).sum(1).sum(1).sum(1).sum(0)
            self.magnitude_statistics[self.activation_index] = values
            self.activation_index += 1
        return y

    def forward(self, x):
        self.activation_index = 0
        self.magnitude_statistics = {}

        def modify_forward(model):
            for child in model.children():
                if is_leaf(child):
                    def new_forward(m):
                        def lambda_forward(x):
                            return self.trace_layer(m, x)
                        return lambda_forward
                    child.old_forward = child.forward
                    child.forward = new_forward(child)
                else:
                    modify_forward(child)

        def restore_forward(model):
            for child in model.children():
                # leaf node
                if is_leaf(child) and hasattr(child, 'old_forward'): 
                    # Update
                    child.forward = child.old_forward
                    child.old_forward = None
                else:
                    restore_forward(child)
        modify_forward(self.model)
        y = self.model.forward(x)
        restore_forward(self.model)
        return y, self.magnitude_statistics
    
if __name__ == "__main__":
    device = 'cuda:0'

    # model
    ori_model = resnet_56()
    ori_model.load_state_dict(torch.load('./model/ckpt/resnet_56_CIFAR10_94.03.t7'))
    ori_model.to(device)
    # slim_channel = [(0, 16), (1, 2), (2, 16), (3, 16), (4, 16), (5, 16), (6, 16), (7, 16), (8, 16), (9, 16), (10, 16), (11, 16), (12, 16), (13, 2), (14, 16), (15, 2), (16, 16), (17, 16), (18, 16), (19, 4), (20, 32), (21, 4), (22, 32), (23, 4), (24, 32), (25, 4), (26, 32), (27, 4), (28, 32), (29, 4), (30, 32), (31, 4), (32, 32), (33, 4), (34, 32), (35, 4), (36, 32), (37, 64), (38, 64), (39, 64), (40, 64), (41, 64), (42, 64), (43, 7), (44, 64), (45, 7), (46, 64), (47, 64), (48, 64), (49, 7), (50, 64), (51, 7), (52, 64), (53, 7), (54, 64)]
    # sparse_channel = [row[1] for row in slim_channel]
    slim_model = torch.load('./ckpt_55/resnet_56_CIFAR10_init.t7').to(device)
    # slim_model = qresnet_56_A(num_classes = 10, filters_left=sparse_channel, bit=[32] * len(sparse_channel)).cuda()
    # slim_model.load_state_dict()
    # slim_model.to(device)

    ori_statistics = BitwidthAllocatorResNet(ori_model)
    slim_statistics = BitwidthAllocatorResNet(slim_model)
    _, ori_magnitude_statistics = ori_statistics.forward(torch.zeros((1,3,32, 32), device = device))
    _, slim_magnitude_statistics = slim_statistics.forward(torch.zeros((1,3,32, 32), device = device))
    layer_sparsity = [slim_magnitude_statistics[i]/ori_magnitude_statistics[i] for i in range(len(ori_magnitude_statistics))]
    qbw_upper = [32] + [8]*18 + [6]*18 + [4]*18
    wbit = [qbw_upper[j] - math.ceil(1/(5*layer_sparsity[j])) for j in range(len(layer_sparsity))]

    print(ori_magnitude_statistics, slim_magnitude_statistics, layer_sparsity)

