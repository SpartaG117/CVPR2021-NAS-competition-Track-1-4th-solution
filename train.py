import utils
import os
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import logging
import torchvision.datasets as dset
import random
from torch.utils.data import DataLoader
import math


CHANNELS_layer1_7 = torch.tensor([4, 8, 12, 16])
CHANNELS_layer8_13 = torch.tensor([4, 8, 12, 16, 20, 24, 28, 32])
CHANNELS_layer14_19 = torch.tensor([4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64])

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--output_dir', type=str, default='4', help='experiment name')
parser.add_argument('--model_name', type=str, default='weights.pth', help='path to save the model')
parser.add_argument('--model_type', type=int, default=0, help="0:weight sharing, 1: operator style")
parser.add_argument('--pretrain_path', type=str, default='', help="pretrain model path")

parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--fixlr', action='store_true', default=False)
parser.add_argument('--lcm', type=int, default=16)


parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--continue_train', action='store_true',
                    help='continue train from a checkpoint')


logger = logging.getLogger('NASTrack1')


def generate_rng(layers, ss_size, lcm, channels):
    # uniform sampling
    rngs = np.ones([layers, lcm], dtype=np.int8)
    for i in range(layers):
        rng_per_layer = list()
        for j in range(math.ceil(lcm / ss_size)):
            if (j + 1) * ss_size - lcm > 0:
                rng_per_layer.extend(np.random.permutation(ss_size).tolist()[:lcm - j * ss_size])
            else:
                rng_per_layer.extend(np.random.permutation(ss_size).tolist())
        rngs[i] = rng_per_layer
    
    rngs = torch.from_numpy(rngs).long().view(-1)
    rngs = torch.gather(channels, 0, rngs).view(layers, lcm)
    return rngs


def generate_rng_2(layers, ss_size, lcm, channels):
    rngs = np.ones([layers, lcm], dtype=np.int8)
    for i in range(layers):
        n_sample = lcm - 2
        assert ss_size - 2 >= n_sample
        if ss_size == lcm:
            rngs[i] = np.random.permutation(ss_size).tolist()
        else:
            idxs = np.concatenate([np.array([0, ss_size - 1]),
                                   np.random.permutation(list(range(1, ss_size - 1)))[:n_sample]],
                                  axis=0)
            np.random.shuffle(idxs)
            rngs[i] = idxs.tolist()
    
    rngs = torch.from_numpy(rngs).long().view(-1)
    rngs = torch.gather(channels, 0, rngs).view(layers, lcm)
    return rngs


def generate_rng_3(layers, ss_size, lcm, channels):
    rngs = np.ones([layers, lcm], dtype=np.int8)
    for i in range(layers):
        if ss_size == lcm:
            rngs[i] = np.random.permutation(ss_size).tolist()
        else:
            idxs = np.concatenate([np.array([0, ss_size - 1, ss_size - 2, ss_size - 3, ss_size - 4]),
                                   np.array([0, ss_size - 1, ss_size - 2, ss_size - 3, ss_size - 4]),
                                   np.random.randint(low=0, high=ss_size, size=[lcm - 10])], axis=0)
            np.random.shuffle(idxs)
            rngs[i] = idxs.tolist()
    
    rngs = torch.from_numpy(rngs).long().view(-1)
    rngs = torch.gather(channels, 0, rngs).view(layers, lcm)
    return rngs


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
    
def train(train_queue, model, criterion, optimizer, args):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    lcm = args.lcm
    total_subnet = lcm * len(train_queue)

    rng1 = generate_rng(7, 4, total_subnet, CHANNELS_layer1_7)
    rng2 = generate_rng(6, 8, total_subnet, CHANNELS_layer8_13)
    rng3 = generate_rng(6, 16, total_subnet, CHANNELS_layer14_19)
    rngs = torch.cat([rng1, rng2, rng3], dim=0).transpose(0, 1).tolist()
    j = 0
    assert j <= total_subnet

    for step, (input, target) in enumerate(train_queue):

        input = input.cuda()
        target = target.cuda()

        optimizer.zero_grad()

        for _ in range(lcm):
            rng = rngs[j]
            j += 1
            logits = model(input, rng)
            loss = criterion(logits, target)
            loss.backward()
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        optimizer.step()

        if step % args.report_freq == 0:
            logger.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion, args):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    rng1 = generate_rng(7, 4, 16, CHANNELS_layer1_7)
    rng2 = generate_rng(6, 8, 16, CHANNELS_layer8_13)
    rng3 = generate_rng(6, 16, 16, CHANNELS_layer14_19)
    rngs = torch.cat([rng1, rng2, rng3], dim=0).transpose(0, 1).tolist()
    rng = rngs[0]
    
    for step, (input, target) in enumerate(zip(valid_queue[0], valid_queue[1])):
        with torch.no_grad():
            input = input.cuda()
            target = target.cuda()
            logits = model(input, rng)
        loss = criterion(logits, target)
    
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
    
        if step % args.report_freq == 0:
            logger.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    return top1.avg, objs.avg


def main(args):
    setup(args)

    if args.model_type == 0:
        from model import ResNet, len_list
        model = ResNet([3, 3, 3], len_list, 100)

    elif args.model_type == 1:
        from model_op import ResNet, len_list
        model = ResNet([3, 3, 3], len_list, 100, args.pretrain_path)


    model.cuda()

    logger.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
        )

    train_transform, valid_transform = utils._data_transforms_cifar100(args)
    train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    
    val_data = torch.from_numpy(np.load(os.path.join(args.data, 'test_inputs.npy')))
    val_labels = torch.from_numpy(np.load(os.path.join(args.data, 'test_labels.npy')))

    infer_bsize = 1000
    val_data = val_data.split(infer_bsize)
    val_labels = val_labels.split(infer_bsize)
    valid_queue = (val_data, val_labels)
    
    if not args.fixlr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    else:
        scheduler = None
    
    start_epoch = 0
    if args.continue_train:
        start_epoch = utils.load(os.path.join(args.output_dir, args.model_name), model, scheduler, optimizer)
        start_epoch += 1
        
    for epoch in range(start_epoch, args.epochs):
        if not args.fixlr:
            # lr = scheduler.get_last_lr()[0]
            lr = utils.get_lr(optimizer)
        else:
            lr = args.learning_rate
        logger.info('epoch %d lr %e', epoch, lr)

        train_acc, train_obj = train(train_queue, model, criterion, optimizer, args)
        logger.info('train_acc %f', train_acc)
        if not args.fixlr:
            scheduler.step()
        
        valid_acc, valid_obj = infer(valid_queue, model, criterion, args)
        logger.info('valid_acc %f', valid_acc)
        
        utils.save(os.path.join(args.output_dir, 'weights.pth'), model, scheduler, epoch, optimizer)
        if epoch % 10 == 0 or epoch >= args.epochs-10:
            utils.save(os.path.join(args.output_dir, 'epoch_{}.pth'.format(epoch)), model, scheduler, epoch, optimizer)


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    main(args)