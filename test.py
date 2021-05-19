import utils
import os
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import logging
import time
import json
from tqdm import tqdm
from scipy.stats import stats
import copy
import random

CHANNELS_layer1_7 = torch.tensor([4, 8, 12, 16])
CHANNELS_layer8_13 = torch.tensor([4, 8, 12, 16, 20, 24, 28, 32])
CHANNELS_layer14_19 = torch.tensor([4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64])

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--output_dir', type=str, default='1', help='experiment name')
parser.add_argument('--model_name', type=str, default='weights.pth', help='path to save the model')
parser.add_argument('--arch', type=str, default='gt_first_1000.json')
parser.add_argument('--no_calibration', action='store_true')
parser.add_argument('--bnbatch', type=int, default=20)
parser.add_argument('--model_type', type=int, default=0, help="0:weight sharing, 1: operator style")


parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')

logger = logging.getLogger('NASTrack1')


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


def generate_rng(layers, ss_size, lcm, channels):
    rngs = np.ones([layers, lcm], dtype=np.int8)
    for i in range(layers):
        rng_per_layer = list()
        for j in range(lcm // ss_size):
            rng_per_layer.extend(np.random.permutation(ss_size).tolist())
        rngs[i] = rng_per_layer
    
    rngs = torch.from_numpy(rngs).long().view(-1)
    rngs = torch.gather(channels, 0, rngs).view(layers, lcm)
    return rngs


def get_eval_rng():
    rng1 = generate_rng(7, 4, 16, CHANNELS_layer1_7)
    rng2 = generate_rng(6, 8, 16, CHANNELS_layer8_13)
    rng3 = generate_rng(6, 16, 16, CHANNELS_layer14_19)
    rngs = torch.cat([rng1, rng2, rng3], dim=0).transpose(0, 1).tolist()
    rng = rngs[0]
    return rng


def infer(model, rng, valid_queue, verbose=True):
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
        
        if step % 100 == 0 and verbose:
            logger.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    if verbose:
        logger.info('valid_acc %f', top1.avg)
    return top1.avg, top5.avg


def setup(args):
    current_rank = 0
    args.output_dir = os.path.join("checkpoints", args.output_dir)
    utils.setup_logger(args.output_dir, distributed_rank=current_rank, name='NASTrack1')
    utils.setup_logger(args.output_dir, distributed_rank=current_rank, name='fvcore')
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
        print(args.output_dir)
    cudnn.benchmark = True
    cudnn.enabled = True
    logger.info("args = %s", args)


def random_eval(valid_queue, train_queue, model):
    rng = get_eval_rng()
    logger.info("rng: {%r}" % rng)
    compute_precise_bn_stats(model, train_queue, rng)
    top1, top5 = infer(model, rng, valid_queue)
    return top1


def gen(args, arch_dict, valid_queue, train_queue, model, submit_dict=None, calibration=True):
    gen_dict = {}
    for k, v in tqdm(arch_dict.items(), total=len(arch_dict)):
        arch = v['arch_int']
        # logger.info("rng: {%r}" % arch)
        if calibration:
            compute_precise_bn_stats(args, model, train_queue, arch)
        top1, top5 = infer(model, arch, valid_queue, False)
        gen_dict[k] = top1 / 100
        if submit_dict is not None:
            submit_dict[k]['acc'] = top1 / 100
    return gen_dict, submit_dict


def compute_precise_bn_stats(args, model, loader, rng, batch_size=128):
    """Computes precise BN stats on training data."""
    # Compute the number of minibatches to use
    model.train()
    num_iter = args.bnbatch
    total = int(num_iter * batch_size)
    start_idx = random.randint(0, len(loader) - total - 1)
    loader = loader[start_idx:start_idx + total]
    loader = loader.split(batch_size)

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
        with torch.no_grad():
            _ = model(inputs.cuda(), rng)
        
        for j, bn in enumerate(bns):
            running_means[j] += bn.running_mean / num_iter
            running_vars[j] += bn.running_var / num_iter
    
    for i, bn in enumerate(bns):
        bn.running_mean = running_means[i]
        bn.running_var = running_vars[i]
        bn.momentum = momentums[i]


def calc_tau_personr(gt, pd):
    gt_acc = []
    pd_acc = []
    for k in gt:
        gt_acc.append(float(gt[k]['acc']))
        pd_acc.append(float(pd[k]))
    corr = stats.kendalltau(gt_acc, pd_acc).correlation
    corr_p = stats.pearsonr(gt_acc, pd_acc)
    # logger.info(gt_acc)
    # logger.info(pd_acc)
    logger.info("tau %r   personr %r" % (corr, corr_p))


def main(args):
    setup(args)
    if args.model_type == 0:
        from model import ResNet, len_list
        model = ResNet([3, 3, 3], len_list)

    elif args.model_type == 1:
        from model_op import ResNet, len_list
        model = ResNet([3, 3, 3], len_list, args.pretrain_path)

    utils.load(os.path.join(args.output_dir, args.model_name), model, None, None, True)
    model.cuda()
    
    logger.info("param size = %fMB", utils.count_parameters_in_MB(model))
    
    train_data = torch.from_numpy(np.load(os.path.join(args.data, 'bn_data_full.npy')))
    val_data = torch.from_numpy(np.load(os.path.join(args.data, 'test_inputs.npy')))
    val_labels = torch.from_numpy(np.load(os.path.join(args.data, 'test_labels.npy')))
    
    infer_bsize = 1000
    val_data = val_data.split(infer_bsize)
    val_labels = val_labels.split(infer_bsize)
    valid_queue = (val_data, val_labels)
    
    arch_dict, ori_dict = parse_json(args)
    out_dict, submit = gen(args, arch_dict, valid_queue, train_data, model, ori_dict,
                           calibration=not args.no_calibration)
    calc_tau_personr(arch_dict, out_dict)



if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)