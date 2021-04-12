import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import sys
from hsd_semantic.utils import *
CUDA_LAUNCH_BLOCKING=1

def find(class_cluster, label):
    for i, C in enumerate(class_cluster):
        if label in C:
            return i
    return -1

def train(train_loader, model, criterion, soft_labels, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    if args.hsd:
        num_clusters = len(model.subnet_classes)
        class_clusters = [tuple(C) for C in model.subnet_classes]
    # switch to train mode
    model.train()

    end = time.time()

    for i,(images, labels) in enumerate(train_loader):
        images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()
        if args.hsd and config.SOLVER.SEPARATE:
            subnet_id = find(class_clusters, labels[0].item())
            outputs = model(images, subnet_id)
            labels, soft_labels = convert(class_clusters[subnet_id], labels, soft_labels)
        else:
            outputs = model(images)
        """
        if args.hsd and config.SOLVER.SEPARATE:
            mask = []
            labels_list = labels.tolist()
            for label in labels_list:
                for i in range(len(model.subnet_classes)):
                    if label in model.subnet_classes[i]:
                        mask.append(model.subnet_mask[i])
                        break
            mask = torch.stack(mask, dim=0).cuda()
            outputs = mask * outputs
        """
        if soft_labels is not None:
            loss = criterion(outputs, labels, soft_labels)
        else:
            loss = criterion(outputs, labels)

        #train_loss += loss.item()
        outputs = F.log_softmax(outputs, dim=1)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, soft_labels, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    if args.hsd:
        num_clusters = len(model.subnet_classes) 
        class_clusters = [tuple(C) for C in model.subnet_classes]
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i,(images, labels) in enumerate(val_loader):
            images, labels = images.cuda() , labels.cuda()
            if args.hsd and config.SOLVER.SEPARATE:
                subnet_id = find(class_clusters, labels[0].item())
                outputs = model(images, subnet_id)
                labels, soft_labels = convert(class_clusters[subnet_id], labels, soft_labels)
            else:
                outputs = model(images)
            """
            if args.hsd and args.separate:
                mask = []
                labels_list = labels.tolist()
                for label in labels_list:
                    for i in range(len(model.subnet_classes)):
                        if label in model.subnet_classes[i]:
                            mask.append(model.subnet_mask[i])
                            break
                mask = torch.stack(mask, dim=0).cuda()
                outputs = mask * outputs
            """
            if soft_labels is not None:
                loss = criterion(outputs, labels, soft_labels)            
            else:
                loss = criterion(outputs, labels)
            outputs = F.log_softmax(outputs, dim=1)
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    return top1.avg, top5.avg

