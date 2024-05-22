import numpy as np
import torch
from torch.backends import cudnn
import tqdm
from model.resnet_cifar import *
from model.vgg_cifar import *

# TODO - 计算模型的推理时间
def cal_runtime(model, optimal_batch_size, device):
    repetitions = 1000

    dummy_input = torch.rand(optimal_batch_size, 3, 32, 32).to(device)

    # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)

    # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    torch.cuda.synchronize()

    # 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # 初始化一个时间容器
    timings = np.zeros((repetitions, 1))

    print('testing ...\n')
    with torch.no_grad():
        for rep in tqdm.tqdm(range(repetitions)):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()  # 等待GPU任务完成
            curr_time = starter.elapsed_time(ender)  # 从 starter 到 ender 之间用时,单位为毫秒
            timings[rep] = curr_time

    avg = timings.sum() / repetitions
    print('\navg_latency={}\n'.format(avg))

    Throughput = (1000*optimal_batch_size)/avg
    print('\navg_Throughput={}\n'.format(Throughput))

    return avg, Throughput

if __name__ == "__main__":

    cudnn.benchmark = True
    device = 'cuda:0'

    # TODO - 计算不同尺寸resnet56的推理时间
    optimal_batch_size_resnet_56 = 512
    model1 = resnet_56(num_classes=10).cuda()
    model1.load_state_dict(torch.load('model/ckpt/resnet_56_CIFAR10_94.03.t7'))
    model1.to(device)
    avg1, Throughput1 = cal_runtime(model1, optimal_batch_size_resnet_56, device)
    speedup = avg1 / avg1
    print('\nspeedup={}X\n'.format(speedup))
    expand_rate = (Throughput1 - Throughput1)*100/Throughput1
    print('\nexpand_rate={}%\n'.format(expand_rate))

    model2 = torch.load('./ckpt_0.03/resnet_56_CIFAR10_init.t7') #pr_30%
    model2.to(device)
    avg2, Throughput2 = cal_runtime(model2, optimal_batch_size_resnet_56, device)
    speedup = avg1 / avg2
    print('\nspeedup={}X\n'.format(speedup))
    expand_rate = (Throughput2 - Throughput1)*100/Throughput1
    print('\nexpand_rate={}%\n'.format(expand_rate))

    model3 = torch.load('/data/cyk/pruning_quant_joint_learning/PIC-PQ-main/ckpt_55/resnet_56_CIFAR10_init.t7') #pr_55%
    model3.to(device)
    avg3, Throughput3 = cal_runtime(model3, optimal_batch_size_resnet_56, device)
    speedup = avg1 / avg3
    print('\nspeedup={}X\n'.format(speedup))
    expand_rate = (Throughput3 - Throughput1)*100/Throughput1
    print('\nexpand_rate={}%\n'.format(expand_rate))

    model4 = torch.load('/data/cyk/pruning_quant_joint_learning/PIC-PQ-main/ckpt_75/resnet_56_CIFAR10_init.t7') #pr_75%
    model4.to(device)
    avg4, Throughput4 = cal_runtime(model4, optimal_batch_size_resnet_56, device)
    speedup = avg1 / avg4
    print('\nspeedup={}X\n'.format(speedup))
    expand_rate = (Throughput4 - Throughput1)*100/Throughput1
    print('\nexpand_rate={}%\n'.format(expand_rate))

    # # TODO - 计算不同尺寸vgg16bn的推理时间
    optimal_batch_size_vgg_16_bn = 256
    model5 = vgg_16_bn().cuda()
    model5.load_state_dict(torch.load('./model/ckpt/vgg_16_bn_CIFAR10_93.75.t7'))
    model5.to(device)
    avg5, Throughput5 = cal_runtime(model5, optimal_batch_size_vgg_16_bn, device)
    speedup = avg5 / avg5
    print('\nspeedup={}X\n'.format(speedup))
    expand_rate = (Throughput5 - Throughput5)*100/Throughput5
    print('\nexpand_rate={}%\n'.format(expand_rate))

    model6 = torch.load('./ckpt_0.025_2/vgg_16_bn_CIFAR10_init.t7') #pr_54%
    model6.to(device)
    avg6, Throughput6 = cal_runtime(model6, optimal_batch_size_vgg_16_bn, device)    
    speedup = avg5 / avg6
    print('\nspeedup={}X\n'.format(speedup))
    expand_rate = (Throughput6 - Throughput5)*100/Throughput5
    print('\nexpand_rate={}%\n'.format(expand_rate))

    model7 = torch.load('/data/cyk/pruning_quant_joint_learning/PIC-PQ-main/ckpt_65.3/vgg_16_bn_CIFAR10_init.t7') #pr_65.3%
    model7.to(device)
    avg7, Throughput7 = cal_runtime(model7, optimal_batch_size_vgg_16_bn, device)
    speedup = avg5 / avg7
    print('\nspeedup={}X\n'.format(speedup))
    expand_rate = (Throughput7 - Throughput5)*100/Throughput5
    print('\nexpand_rate={}%\n'.format(expand_rate))

    model8 = torch.load('./ckpt_0.01/vgg_16_bn_CIFAR10_init.t7') #pr_75%
    model8.to(device)
    avg8, Throughput8 = cal_runtime(model8, optimal_batch_size_vgg_16_bn, device)
    speedup = avg5 / avg8
    print('\nspeedup={}X\n'.format(speedup))
    expand_rate = (Throughput8 - Throughput5)*100/Throughput5
    print('\nexpand_rate={}%\n'.format(expand_rate))

    

  # filter_pruner = eval('FilterPrunerResNet')(model, 'Rank', num_cls=10, rankPath='./rank_conv/CIFAR10/resnet_56', device=device)
    # filter_pruner.reset() 
    # model.eval()
    # filter_pruner.forward(torch.zeros((1,3,32,32), device=device))
    # perturbation = np.loadtxt('./log_0.018/resnet_56_CIFAR10_ea_min.data')
    # sparse_channel, bit, cur_bops = filter_pruner.pruning_with_transformations(filter_pruner.filter_ranks, perturbation, flops_target = 50, bops_target = 0.018)
    # slim_channel = [(0, 16), (1, 16), (2, 16), (3, 16), (4, 16), (5, 5), (6, 16), (7, 16), (8, 16), (9, 2), (10, 16), (11, 2), (12, 16), (13, 2), (14, 16), (15, 2), (16, 16), (17, 16), (18, 16), (19, 32), (20, 32), (21, 32), (22, 32), (23, 4), (24, 32), (25, 32), (26, 32), (27, 4), (28, 32), (29, 4), (30, 32), (31, 32), (32, 32), (33, 4), (34, 32), (35, 32), (36, 32), (37, 7), (38, 64), (39, 7), (40, 64), (41, 7), (42, 64), (43, 7), (44, 64), (45, 64), (46, 64), (47, 7), (48, 64), (49, 7), (50, 64), (51, 64), (52, 64), (53, 7), (54, 64)]
    # sparse_channel = [row[1] for row in slim_channel]
    # bit = [8, 2, 8, 2, 8, 8, 8, 2, 8, 2, 8, 2, 8, 2, 8, 2, 8, 2, 8, 2, 8, 2, 8, 2, 8, 2, 8, 2, 8, 2, 8, 2, 8, 2, 8, 2, 8, 2, 8, 2, 8, 2, 8, 2, 8, 2, 8, 2, 8, 2, 8, 2, 8, 2, 8]
    # sparse_channel = [16]*19 + [32]*18 + [64]*18
    # bit = [32]*55
    # model = slim_resnet_56(num_classes=10, filters_left=sparse_channel).cuda()
    # model.load_state_dict(torch.load('./ckpt_0.018_1/resnet_56_CIFAR10_best.t7'))