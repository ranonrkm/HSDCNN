import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from hsd_semantic.datasets import get_CifarDataLoader, get_TImageNetLoader
from hsd_semantic.models.imagenet import get_network
from hsd_semantic.models.subnets import get_hsd
from hsd_semantic.hierarchy import get_Softlabels
from hsd_semantic.hierarchy.trees import load_distances, load_hierarchy 
from hsd_semantic.hierarchy.labels import make_all_soft_labels
from hsd_semantic.tools import *
from hsd_semantic.utils import *
from hsd_semantic.config import config

import time
import os
import sys
import argparse
from datetime import date
today = date.today()
CUDA_LAUNCH_BLOCKING=1

best_acc1 = 0
best_acc5 = 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-H', '--hsd', dest='hsd', action='store_true',
                    help='train or evaluate hsd model')
    parser.add_argument('-S', '--separate', dest='separate', action='store_true',
                    help='train subnetworks alltogether ot separately')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-P', '--pretrained',dest='pretrained', action='store_true',
                    help='use pretrained network')
    args = parser.parse_args()

    if args.hsd == False:
        print('Undecomposed Network')
        net = get_network(args)
        if args.pretrained:
            try:
                net.load_state_dict(torch.load(config.MODEL.CKPT))
                print('using pretrained network.....')
            except:
                print('can not load pretrained network')
                pass

    else:
        print('hsd')
        net = get_hsd()
        try:
            net.load_state_dict(torch.load(config.MODEL.HSD_CKPT))
        except FileNotFoundError:
            pass

        # TODO: Add code for masking out subnetworks based on classes of interest

    net = net.cuda()

    train_loader, val_loader = get_TImageNetLoader()
    print("################Data-sets loaded####################\n")

    # Load hierarchy and classes ------------------------------------------------------------------------------------------------------------------------------
    distances = load_distances(config.DATASET.NAME, 'ilsvrc', config.HIERARCHY.ROOT)
    hierarchy = load_hierarchy(config.DATASET.NAME)
    classes = train_loader.dataset.classes

    if config.SOLVER.LOSS == "hierarchical-cross-entropy":
        weights = get_weighting(hierarchy, "exponential", value=config.HIERARCHY.ALPHA)
        loss_function = HierarchicalCrossEntropyLoss(hierarchy, classes, weights).cuda()
    elif config.SOLVER.LOSS == "soft-labels":
        loss_function = nn.KLDivLoss().cuda()
    else:
        loss_function = nn.CrossEntropyLoss().cuda()
    print('using {} loss function'.format(config.SOLVER.LOSS))
   
    soft_labels = make_all_soft_labels(distances, classes, config.HIERARCHY.BETA)
 
    optimizer = optim.SGD(net.parameters(),
                          lr=config.SOLVER.BASE_LR,
                          momentum=config.SOLVER.MOMENTUM,
                          weight_decay=config.SOLVER.WEIGHT_DECAY)
    LR_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                  factor=config.SOLVER.SCHEDULER_FACTOR,
                                                  patience=config.SOLVER.SCHEDULER_PATIENCE)
    global best_acc1
    global best_acc5
    if args.evaluate:
        validate(val_loader, net, criterion, args)
        sys.exit(0)
        #return

    for epoch in range(config.SOLVER.NUM_EPOCHS):
        #adjust_learning_rate(optimizer, epoch, args)
        for param_group in optimizer.param_groups:
                print("learning rate:",param_group['lr'])

        # train for one epoch
        train(train_loader, net, loss_function, distances, soft_labels, classes, optimizer, epoch, args)
        # evaluate on validation set
        acc1,acc5 = validate(val_loader, net, loss_function, soft_labels, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        LR_scheduler.step(best_acc1)
 
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': config.MODEL.NAME,
            'state_dict': net.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
        for param_group in optimizer.param_groups:
                print("learning rate:",param_group['lr'])
                if param_group['lr'] < 1e-12:
                    print(epoch)
                    print("------------training ends-------------")
                    sys.exit(0)



if __name__ == '__main__':
    main()
