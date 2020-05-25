import argparse
import os
import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import *
from models import Perceptron
from utils import *
from torchvision.transforms import Compose


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/KTH/')
parser.add_argument('--key_joints', type=list, default=[1, 2, 3, 4, 5, 6, 7], help='which dimensions to use')

parser.add_argument('--input_dim', type=int, default=14, help='watch out the consistency with key_joints')
parser.add_argument('--mid_dim', type=int, default=20)
parser.add_argument('--class_num', type=int, default=4)

parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--epochs', default=50, type=int)

parser.add_argument('--report_freq', default=50, type=int)

args = parser.parse_args()


def main(args):
    model = Perceptron(args.input_dim, args.mid_dim, args.class_num)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.5, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    transforms = Compose([PoseNormalize(0), ToTensor()])
    train_set = KeyPointDataset(args.data_dir + 'data_final.json', args.key_joints, transform=transforms)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

    for epoch in range(args.epochs):
        scheduler.step()
        print('Epoch {}'.format(epoch))
        top1, top5 = train(model, optimizer, train_loader, criterion, args.report_freq)
        print('{} {}'.format(top1, top5))


def train(model, optimizer, train_loader, criterion, report_freq):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for step, sample in enumerate(train_loader):
        input = sample['key_point']
        label = sample['label']
        batch_size = input.shape[0]

        input = input.float()
        label = label.long()

        output = model(input)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(output, label, topk=(1, 1))
        objs.update(loss.item(), batch_size)
        top1.update(prec1.item(), batch_size)
        top5.update(prec5.item(), batch_size)

        if step % report_freq == 0:
            print('Train Step {} Loss {} Top1 {} Top5 {}'.format(step, objs.avg, top1.avg, top5.avg))

    return top1.avg, top5.avg


if __name__ == '__main__':
    main(args)
