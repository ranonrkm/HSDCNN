import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from hsd_semantic.datasets import get_CifarDataLoader
from hsd_semantic.models import get_network
from hsd_semantic.models.subnets import get_hsd
from hsd_semantic.utils import progress_bar, WarmUpLR, summary
from hsd_semantic.hierarchy import get_Softlabels
from hsd_semantic.config import config

import time
import os
import sys
import argparse
from datetime import date
today = date.today()
CUDA_LAUNCH_BLOCKING=1


def loss_function(pred, target):
    y = np.zeros(pred.size())
    for i in range(len(target)):
        label = target[i]
        y[i] = soft_labels[label]
    y = torch.FloatTensor(y).to(pred.device)
    pred = F.log_softmax(pred, dim=1)
    loss = y * pred
    loss = -1. * loss.sum(-1).mean()
    return loss

def train(epoch):
    net.train()
    train_loss = 0.9
    total = 0
    correct = 0
    for batch_idx,(images, labels) in enumerate(train_loader):
        images, labels = images.cuda(), labels.cuda()
        
        optimizer.zero_grad()
        outputs = net(images)
        if args.hsd and config.SOLVER.SEPARATE:
            mask = []
            labels_list = labels.tolist()
            for label in labels_list:
                for i in range(len(subnet_cls)):
                    if label in subnet_cls[i]:
                        mask.append(subnet_mask[i])
                        break
            mask = torch.stack(mask, dim=0).cuda()
            outputs = mask * outputs

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        outputs = F.log_softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                                    % (train_loss/(batch_idx+1),
                                                    100.*correct/total, correct, total))
    tra_acc = 100.*float(correct)/total
    return tra_acc 

def eval_training(epoch):
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0
    total = 0
    lambda_ = 1/(epoch+1) #5e-6

    for batch_idx,(images, labels) in enumerate(test_loader):

        images, labels = images.cuda() , labels.cuda()

        outputs = net(images)
        if args.hsd and config.SOLVER.SEPARATE:
            mask = []
            labels_list = labels.tolist()
            for label in labels_list:
                for i in range(len(subnet_cls)):
                    if label in subnet_cls[i]:
                        mask.append(subnet_mask[i])
                        break
            mask = torch.stack(mask, dim=0).cuda()
            outputs = mask * outputs
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum()
        progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                                   % (test_loss/(batch_idx+1),
                                                   100.*correct/total, correct, total))
    return 100.*float(correct)/total, test_loss/total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-H', '--hsd', dest='hsd', action='store_true',
                    help='train or evaluate hsd model')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
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
    
    train_loader, test_loader = get_CifarDataLoader()
    print("################Data-sets loaded####################\n")
   
    if config.HIERARCHY.BETA >= 100 or config.SOLVER.XE:
        print("Standard CE loss\n")
        loss_function = nn.CrossEntropyLoss()
    else: 
        # Hierarchical cross entropy loss
        # Using soft labels
        print("Using Soft labels for cross-entropy loss\n")
        hierarchy_file = os.path.join(config.HIERARCHY.ROOT, config.DATASET.NAME + '-Hierarchy', 'parent-child.txt')
        class_list = os.path.join(config.HIERARCHY.ROOT, config.DATASET.NAME + '-Hierarchy', 'class_names.txt')
        soft_labels = get_Softlabels(hierarchy_file, config.HIERARCHY.BETA, class_list)[:config.DATASET.NUM_CLASSES, :config.DATASET.NUM_CLASSES]
    
    optimizer = optim.SGD(net.parameters(), 
                          lr=config.SOLVER.BASE_LR,
                          momentum=config.SOLVER.MOMENTUM,
                          weight_decay=config.SOLVER.WEIGHT_DECAY)
    LR_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                  factor=config.SOLVER.SCHEDULER_FACTOR, 
                                                  patience=config.SOLVER.SCHEDULER_PATIENCE)

    checkpoint_new = os.path.join(config.MODEL.CKPT_PATH, '{net}-{date}-{beta}-{type}.pth')
    best_acc = 0.0
    best_loss = 999.0
    eval_training(0)
    if args.evaluate:
        sys.exit(0)
   
    train_end = False
    for epoch in range(1, config.SOLVER.NUM_EPOCHS):
        if train_end:
            break
        LR_scheduler.step(best_acc)
        train_acc = train(epoch)
        acc, test_loss = eval_training(epoch)
        #start to save best performance model after learning rate decay to 0.01
        if best_loss > test_loss:
            torch.save(net.state_dict(),
                       checkpoint_new.format(net=config.MODEL.NAME,
                                             date=today.strftime('%d')+today.strftime('%m'),
                                             beta=config.HIERARCHY.BETA,
                                             type='loss'))
            best_loss = test_loss
        if best_acc < acc:
            torch.save(net.state_dict(),
                       checkpoint_new.format(net=config.MODEL.NAME,
                                             date=today.strftime('%d')+today.strftime('%m'),
                                             beta=config.HIERARCHY.BETA,
                                             type='acc'))
            best_acc = acc
            continue
        #print("best_acc: %.3f\ttest_acc: %.3f\ttrain_acc: %.3f"%(best_acc,acc,tra_acc))
        for param_group in optimizer.param_groups:
            print("learning rate:",param_group['lr'])
            if param_group['lr'] < 1e-10:
                print(epoch)
                print("------------training ends-------------\n")
                train_end = True
   
