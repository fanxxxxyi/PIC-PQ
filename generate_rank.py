import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from utils import *
from model.drives import *

from model.resnet_cifar import *
from model.vgg_cifar import *
from model.MobileNetV2 import *

from pruner.filterpruner import FilterPruner
from pruner.fp_resnet import FilterPrunerResNet
from pruner.fp_vgg import FilterPrunerVGG

os.environ['CUDA_VISIBLE_DEVICES']='0'
# Hyper-Parameters
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--arch',
        type=str,
        default='vgg_16_bn',
        choices=('vgg_16_bn', 'resnet_56', 'mobilenetV2'),
        help='The architecture to prune and the resulting model and logs will use this')
    parser.add_argument(
        '--resume',
        type=str,
        default='./ckpt/vgg_16_bn_CIFAR10.t7',
        help='load the model from the specified checkpoint') # default=None
    parser.add_argument(
        "--datapath",
        type=str,
        default='./data',
        help='Path toward the dataset that is used for this experiment')
    parser.add_argument(
        "--dataset",
        type=str,
        default='torchvision.datasets.CIFAR10',
        help='The class name of the dataset that is used, please find available classes under the dataset folder')
    parser.add_argument(
        "--pruner",
        type=str,
        default='FilterPrunerVGG',
        choices=('FilterPrunerResNet', 'FilterPrunerVGG', 'FilterPrunerMBNetV2'),
        help='Different network require differnt pruner implementation')
    parser.add_argument(
        "--rank_type",
        type=str,
        default='Rank',
        choices=('l1_bn','l2_bn','l1_weight','l2_weight', 'Rank'),
        help='The ranking criteria for filter pruning')
    parser.add_argument(
        "--global_random_rank",
        action='store_true',
        default=False,
        help='When this is specified, none of the rank_type matters, it will randomly prune the filters')
    parser.add_argument(
        "--safeguard",
        type=float,
        default=0,
        help='A floating point number that represent at least how many percentage of the original number of channel should be preserved. E.g., 0.10 means no matter what ranking, each layer should have at least 10% of the number of original channels.')
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        choices=(16, 32, 64, 128),
        help='Batch size for training.')
    parser.add_argument(
        '--img_size',
        type=int,
        default=32,
        help='32 is the size of CIFAR10')
    parser.add_argument(
        "--no_val",
        action='store_true',
        default=False,
        help='Use full dataset to train (use to compare with prior art in CIFAR-10)')
    parser.add_argument(
        "--gpu",
        type=str,
        default='0',
        help='Select GPU to use.("0" is to use GPU:0))')
    parser.add_argument(
        '--limit',
        type=int,
        default=6,
        help='The number of  batch to get rank.')

    args = parser.parse_args()
    return args


# NOTE hook function
criterion = nn.CrossEntropyLoss()
feature_result = torch.tensor(0.)
total = torch.tensor(0)
def get_feature_hook(self, input, output):
    global feature_result
    global total

    a = output.shape[0]
    b = output.shape[1]
    c = torch.tensor([torch.matrix_rank(output[i, j, :, :]).item() for i in range(a) for j in range(b)])

    c = c.view(a, -1).float()
    c = c.sum(0)
    feature_result = feature_result * total + c
    total = total + a
    feature_result = feature_result / total


def get_feature_hook_densenet(self, input, output):
    global feature_result
    global total
    a = output.shape[0]
    b = output.shape[1]
    c = torch.tensor([torch.matrix_rank(output[i, j, :, :]).item() for i in range(a) for j in range(b-12,b)])

    c = c.view(a, -1).float()
    c = c.sum(0)
    feature_result = feature_result * total + c
    total = total + a
    feature_result = feature_result / total


def inReference():
    model.eval()
    temp_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx >= args.limit:  # use the first args.limit+1 batches to estimate the rank
                break

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            temp_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, args.limit, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (temp_loss/(batch_idx+1), 100.*correct/total, correct, total))


# ----------- start -----------------
if __name__ == "__main__":
    # Parameters
    startTime = time.time()
    args = get_args()
    device = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
    print("Environment：\ndevice --- {} \ntorch --- {} \ntorchvision --- {} \n\nParameters:\n{}\n".format(device,torch.__version__, torchvision.__version__, args))

    # Num_classes
    if 'CIFAR100' in args.dataset:
        num_classes = 100
    elif 'CIFAR10' in args.dataset:
        num_classes = 10
    elif 'ImageNet' in args.dataset:
        num_classes = 1000
    elif 'CUB200' in args.dataset:
        num_classes = 200

    #Data
    train_loader, val_loader, test_loader = get_dataloader(args.img_size, args.dataset, args.datapath, args.batch_size, args.no_val)

    # model
    model = vgg_16_bn().cuda()
    model.load_state_dict(torch.load(args.resume))
    # model = torch.load(args.resume, map_location=device)
    model = model.to(device)
    print(model)
    
    # Pruner
    pruner = eval(args.pruner)(model, args.rank_type, num_classes, args.safeguard, random=args.global_random_rank, device=device)
    model.train()

    # NOTE Get ranks of feature map
    if args.arch == 'vgg_16_bn':
        # handle directory
        if 'CIFAR100' in args.dataset:
            if not os.path.isdir('rank_conv/CIFAR100/{}/'.format(args.arch)):
                os.makedirs('rank_conv/CIFAR100/{}/'.format(args.arch))
        elif 'CIFAR10' in args.dataset:
            if not os.path.isdir('rank_conv/CIFAR10/{}/'.format(args.arch)):
                    os.makedirs('rank_conv/CIFAR10/{}/'.format(args.arch))
        
        ''' Obtain rank '''
        for i, cov_id in enumerate(model.covcfg):
            cov_layer = model.features[cov_id]
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inReference()
            handler.remove()

            if 'CIFAR100' in args.dataset:
                np.save('rank_conv/CIFAR100/{}/'.format(args.arch) + '/rank_conv%d' % (i + 1) + '.npy', feature_result.numpy())
            elif 'CIFAR10' in args.dataset:
                np.save('rank_conv/CIFAR10/{}/'.format(args.arch) + '/rank_conv%d' % (i + 1) + '.npy', feature_result.numpy())

            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

    elif args.arch == 'resnet_56':
        # handle directory
        if 'CIFAR100' in args.dataset:
            if not os.path.isdir('rank_conv/CIFAR100/{}/'.format(args.arch)):
                os.makedirs('rank_conv/CIFAR100/{}/'.format(args.arch))
        elif 'CIFAR10' in args.dataset:
            if not os.path.isdir('rank_conv/CIFAR10/{}/'.format(args.arch)):
                    os.makedirs('rank_conv/CIFAR10/{}/'.format(args.arch))
        
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)
        
        # First conv layer
        cov_layer = eval('model.conv1')
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inReference()
        handler.remove()
        if 'CIFAR100' in args.dataset:
            np.save('rank_conv/CIFAR100/{}/'.format(args.arch) + '/rank_conv%d' % (1) + '.npy', feature_result.numpy())
        elif 'CIFAR10' in args.dataset:
            np.save('rank_conv/CIFAR10/{}/'.format(args.arch) + '/rank_conv%d' % (1) + '.npy', feature_result.numpy())

        # 
        cnt = 1
        for i in range(3):
            # get block
            block = eval('model.layer%d' % (i + 1))

            for j in range(9):
                # cov_layer = block[j].relu1
                cov_layer = block[j].conv1
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inReference()
                handler.remove()
                if 'CIFAR100' in args.dataset:
                    np.save('rank_conv/CIFAR100/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
                elif 'CIFAR10' in args.dataset:
                    np.save('rank_conv/CIFAR10/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
            

                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                ''' * '''
                cov_layer = block[j].conv2
                # cov_layer = block[j].relu2
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inReference()
                handler.remove()
                if 'CIFAR100' in args.dataset:
                    np.save('rank_conv/CIFAR100/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
                elif 'CIFAR10' in args.dataset:
                    np.save('rank_conv/CIFAR10/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())

                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

    elif args.arch == 'mobilenetV2':
        # handle directory
        if 'CIFAR100' in args.dataset:
            if not os.path.isdir('rank_conv/CIFAR100/{}/'.format(args.arch)):
                os.makedirs('rank_conv/CIFAR100/{}/'.format(args.arch))
        elif 'CIFAR10' in args.dataset:
            if not os.path.isdir('rank_conv/CIFAR10/{}/'.format(args.arch)):
                    os.makedirs('rank_conv/CIFAR10/{}/'.format(args.arch))
            
        # first Conv
        cov_layer = eval('model.features[0][2]')
        # print(str(cov_layer) + ': 1')
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inReference()
        handler.remove()

        if 'CIFAR100' in args.dataset:
            np.save('rank_conv/CIFAR100/{}/'.format(args.arch) + '/rank_conv%d' % (1) + '.npy', feature_result.numpy())
        elif 'CIFAR10' in args.dataset:
            np.save('rank_conv/CIFAR10/{}/'.format(args.arch) + '/rank_conv%d' % (1) + '.npy', feature_result.numpy())
        
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

        # InvertedResidual block
        cnt = 1
        for i in range(17):
            # First invertedResidual
            if i < 1:
                # First conv layer
                cov_layer = eval('model.features[%d].conv[0]' %(i + 1))
                # print(str(cov_layer) + ': %d' %(cnt + 1))
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inReference()
                handler.remove()

                if 'CIFAR100' in args.dataset:
                    np.save('rank_conv/CIFAR100/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
                elif 'CIFAR10' in args.dataset:
                    np.save('rank_conv/CIFAR10/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())

                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)
                cnt += 1

                # Second conv layer
                cov_layer = eval('model.features[%d].conv[3]' %(i + 1))
                # print(str(cov_layer) + ': %d' %(cnt + 1))
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inReference()
                handler.remove()

                if 'CIFAR100' in args.dataset:
                    np.save('rank_conv/CIFAR100/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
                elif 'CIFAR10' in args.dataset:
                    np.save('rank_conv/CIFAR10/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
                                
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)
                cnt += 1
                
            # 2~17 invertedResidual
            else:
                # First conv layer
                cov_layer = eval('model.features[%d].conv[0]' %(i + 1))
                # print(str(cov_layer) + ': %d' %(cnt + 1))
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inReference()
                handler.remove()

                if 'CIFAR100' in args.dataset:
                    np.save('rank_conv/CIFAR100/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
                elif 'CIFAR10' in args.dataset:
                    np.save('rank_conv/CIFAR10/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
                
                
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)
                cnt += 1

                # Second conv layer
                cov_layer = eval('model.features[%d].conv[3]' %(i + 1))
                # print(str(cov_layer) + ': %d' %(cnt + 1))
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inReference()
                handler.remove()

                if 'CIFAR100' in args.dataset:
                    np.save('rank_conv/CIFAR100/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
                elif 'CIFAR10' in args.dataset:
                    np.save('rank_conv/CIFAR10/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
                
                
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)
                cnt += 1

                # Third conv layer
                cov_layer = eval('model.features[%d].conv[6]' %(i + 1))
                # print(str(cov_layer) + ': %d' %(cnt + 1))
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inReference()
                handler.remove()

                if 'CIFAR100' in args.dataset:
                    np.save('rank_conv/CIFAR100/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
                elif 'CIFAR10' in args.dataset:
                    np.save('rank_conv/CIFAR10/{}/'.format(args.arch) + '/rank_conv%d' % (cnt + 1) + '.npy', feature_result.numpy())
        
                
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)
                cnt += 1

        # Last Conv e.g., 52th
        cov_layer = eval('model.features[18][2]')
        # print(str(cov_layer) + ': %d' %(cnt + 1))
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inReference()
        handler.remove()

        if 'CIFAR100' in args.dataset:
            np.save('rank_conv/CIFAR100/{}/'.format(args.arch) + '/rank_conv%d' % (52) + '.npy', feature_result.numpy())
        elif 'CIFAR10' in args.dataset:
            np.save('rank_conv/CIFAR10/{}/'.format(args.arch) + '/rank_conv%d' % (52) + '.npy', feature_result.numpy())
        
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

    print('\n----------- Cost Time: ' + format_time(time.time() - startTime) +" ----------- \n----------- Program Over -----------")
