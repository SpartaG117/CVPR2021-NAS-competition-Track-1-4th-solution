from model_op import ResNet, len_list, BasicBlock

import utils
import os
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import json
import copy
import time
import math
import random

CHANNELS_layer1_7 = torch.tensor([4, 8, 12, 16])
CHANNELS_layer8_13 = torch.tensor([4, 8, 12, 16, 20, 24, 28, 32])
CHANNELS_layer14_19 = torch.tensor([4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64])

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--output_dir', type=str, default='op10_3', help='experiment name')
parser.add_argument('--model_name', type=str, default='weights.pth', help='path to save the model')
parser.add_argument('--arch', type=str, default='Track1_final_archs.json')

# parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=0)

parser.add_argument('--num_gpu', type=int, default=1, help='gpu device id')
parser.add_argument('--proc_id', type=int, default=0)
parser.add_argument('--n_p', type=int, default=12)


def div_remainder(n, interval):
    # finds divisor and remainder given some n/interval
    factor = math.floor(n / interval)
    remainder = int(n - (factor * interval))
    return factor, remainder


def show_time(seconds):
    # show amount of time as human readable
    if seconds < 60:
        return "{:.2f}s".format(seconds)
    elif seconds < (60 * 60):
        minutes, seconds = div_remainder(seconds, 60)
        return "{}m,{}s".format(minutes, seconds)
    else:
        hours, seconds = div_remainder(seconds, 60 * 60)
        minutes, seconds = div_remainder(seconds, 60)
        return "{}h,{}m,{}s".format(hours, minutes, seconds)
    
def parse_json(args):
    with open(args.arch, 'r') as f:
        data = json.load(f)
    
    ori = copy.deepcopy(data)
    for k, v in data.items():
        arch = v['arch']
        arch = arch.split('-')
        assert len(arch) == 20
        arch = [int(a) for a in arch[:-1]]
        data[k]['arch_int'] = arch
    return data, ori


def infer(model, rng, valid_queue):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    
    for step, (input, target) in enumerate(zip(valid_queue[0], valid_queue[1])):
        with torch.no_grad():
            input = input.cuda()
            target = target.cuda()
            logits = model(input, rng)
        
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        
    return top1.avg, top5.avg


def setup(args):
    current_rank = 0
    args.output_dir = os.path.join("checkpoints", args.output_dir)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
        print(args.output_dir)
    
    cudnn.benchmark = True
    cudnn.enabled = True


def gen(proc_id, arch_dict, valid_queue, train_queue, model, submit_dict=None):
    gen_dict = {}
    num = len(arch_dict)
    start_time = time.time()
    for i, (k, v) in enumerate(arch_dict.items()):
        arch = v['arch_int']
        compute_precise_bn_stats(model, train_queue, arch)
        top1, top5 = infer(model, arch, valid_queue)
        gen_dict[k] = top1 / 100
        if submit_dict is not None:
            submit_dict[k]['acc'] = top1 / 100
        if i % 10 == 0:
            average_epoch_t = (time.time() - start_time) / (i + 1)
            print("process {} handle {}/{}  etc:{}" .format(proc_id, i, num, show_time(average_epoch_t * (num - i - 1))))
            
    return gen_dict, submit_dict


def compute_precise_bn_stats(model, loader, rng, batch_size=128):
    """Computes precise BN stats on training data."""
    # Compute the number of minibatches to use
    model.train()
    num_iter = 50
    total = int(num_iter * batch_size)
    start_idx = random.randint(0, len(loader) - total - 1)
    loader = loader[start_idx:start_idx+total]
    loader = loader.split(batch_size)

    # Retrieve the BN layers
    bns = model.get_bn(rng)

    # Initialize BN stats storage for computing mean(mean(batch)) and mean(var(batch))
    running_means = [torch.zeros_like(bn.running_mean) for bn in bns]
    running_vars = [torch.zeros_like(bn.running_var) for bn in bns]
    # Remember momentum values
    momentums = [bn.momentum for bn in bns]
    # Set momentum to 1.0 to compute BN stats that only reflect the current batch
    for bn in bns:
        bn.momentum = 1.0
    # Average the BN stats for each BN layer over the batches
    for inputs in loader:
        _ = model(inputs.cuda(), rng)
        
        for j, bn in enumerate(bns):
            running_means[j] += bn.running_mean / num_iter
            running_vars[j] += bn.running_var / num_iter
    
    for i, bn in enumerate(bns):
        bn.running_mean = running_means[i]
        bn.running_var = running_vars[i]
        bn.momentum = momentums[i]


def single_core(proc_id, args, arch_dict_n, ori_dict_n):
    cudnn.benchmark = True
    cudnn.enabled = True
    n_gpu = torch.cuda.device_count()
    proc_per_gpu = args.n_p // n_gpu
    torch.cuda.set_device(proc_id//proc_per_gpu)
    model = ResNet([3, 3, 3], len_list)
    utils.load(os.path.join(args.output_dir, args.model_name), model, None, None, True)

    train_data = torch.from_numpy(np.load(os.path.join(args.data, 'bn_data_full.npy')))
    val_data = torch.from_numpy(np.load(os.path.join(args.data, 'test_inputs.npy')))
    val_labels = torch.from_numpy(np.load(os.path.join(args.data, 'test_labels.npy')))

    infer_bsize = 1000
    val_data = val_data.split(infer_bsize)
    val_labels = val_labels.split(infer_bsize)
    valid_queue = (val_data, val_labels)
    
    model.cuda()

    _, submit = gen(proc_id, arch_dict_n, valid_queue, train_data, model, ori_dict_n)
    
    with open(os.path.join(args.output_dir, 'submission','submit'+str(proc_id)+'.json'), 'w') as f:
        json.dump(submit, f)
    return submit
    
    
def main(args):
    setup(args)
    arch_dict, ori_dict = parse_json(args)
    n_p = args.n_p
    p_id = args.proc_id
    idx_list = np.arange(len(ori_dict.keys()))
    if args.start != args.end:
        idx_list = idx_list[args.start:args.end]
    split_key_index = np.array_split(idx_list, n_p)

    key_list = split_key_index[p_id].tolist()
    arch_dict_n = {}
    ori_dict_n = {}
    keys = list(ori_dict.keys())
    for i in key_list:
        k = keys[i]
        arch_dict_n[k] = arch_dict[k]
        ori_dict_n[k] = ori_dict[k]
    
    single_core(args.proc_id, args, arch_dict_n, ori_dict_n)


if __name__ == '__main__':
    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)