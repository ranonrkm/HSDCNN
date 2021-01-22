import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from hsd_semantic.datasets import get_CifarDataLoader
from hsd_semantic.models import get_network
from hsd_semantic.models.subnets import get_hsd
from hsd_semantic.hierarchy import get_Softlabels
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
    args = parser.parse_args()

    if args.hsd == False:
        print('Undecomposed Network')
        net = get_network()
        try:
            net.load_state_dict(torch.load(config.MODEL.CKPT))
        except FileNotFoundError:
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

    train_loader, val_loader = get_CifarDataLoader()
    print("################Data-sets loaded####################\n")

    if config.HIERARCHY.BETA >= 100 or config.SOLVER.XE:
        print("Standard CE loss\n")
        criterion = nn.CrossEntropyLoss()
    else:
        # Hierarchical cross entropy loss
        # Using soft labels
        print("Using Soft labels for cross-entropy loss\n")
        hierarchy_file = os.path.join(config.HIERARCHY.ROOT, config.DATASET.NAME + '-Hierarchy', 'parent-child.txt')
        class_list = os.path.join(config.HIERARCHY.ROOT, config.DATASET.NAME + '-Hierarchy', 'class_names.txt')
        soft_labels = get_Softlabels(hierarchy_file, config.HIERARCHY.BETA, class_list)[:config.DATASET.NUM_CLASSES, :config.DATASET.NUM_CLASSES]
        criterion = losses.loss_function

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
        train(train_loader, net, criterion, soft_labels, optimizer, epoch, args)
        # evaluate on validation set
        acc1,acc5 = validate(val_loader, net, criterion, soft_labels, args)

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
