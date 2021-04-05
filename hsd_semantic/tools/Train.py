import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import sys
from hsd_semantic.hierarchy.labels import make_batch_soft_labels, make_batch_onehot_labels
from hsd_semantic.utils import *
CUDA_LAUNCH_BLOCKING=1


def train(train_loader, model, loss_function, distances, all_soft_labels, classes, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    max_dist = max(distances.distances.values())
    best_hier_similarities = _make_best_hier_similarities(classes, distances, max_dist)
    """ 
    loss_accum = 0.0
    norm_mistakes_accum = 0.0
    flat_accuracy_accums = np.zeros(len(topK_to_consider), dtype=np.float)
    hdist_accums = np.zeros(len(topK_to_consider))
    hdist_top_accums = np.zeros(len(topK_to_consider))
    hdist_mistakes_accums = np.zeros(len(topK_to_consider))
    hprecision_accums = np.zeros(len(topK_to_consider))
    hmAP_accums = np.zeros(len(topK_to_consider)) 
    """
    # switch to train mode
    model.train()

    end = time.time()

    for i,(images, labels) in enumerate(train_loader):
        images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(images)
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
        if config.SOLVER.LOSS == 'soft-labels':
            outputs = F.log_softmax(outputs, dim=1)
            target_distribution = make_batch_soft_labels(all_soft_labels, labels,
                                    config.DATASET.NUM_CLASSES, config.SOLVER.BATCH_SIZE)
        else:
            target_distribution = make_batch_onehot_labels(labels, 
                                    config.DATASET.NUM_CLASSES, config.SOLVER.BATCH_SIZE)
        loss = loss_function(outputs, target_distribution)
        if config.SOLVER.LOSS == 'cross-entropy':
            outputs = F.log_softmax(outputs, dim=1)
        #train_loss += loss.item()
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


def validate(val_loader, model, loss_function, all_soft_labels, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    max_dist = max(distances.distances.values())
    best_hier_similarities = _make_best_hier_similarities(classes, distances, max_dist)

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i,(images, labels) in enumerate(val_loader):

            images, labels = images.cuda() , labels.cuda()

            outputs = model(images)
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
            if config.SOLVER.LOSS == 'soft-labels':
                outputs = F.log_softmax(outputs, dim=1)
                target_distribution = make_batch_soft_labels(all_soft_labels, labels)
            else:
                target_distribution = make_batch_onehot_labels(labels)
            losses = loss_function(outputs, target_distribution)
            if config.SOLVER.LOSS == 'cross-entropy':
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


def _make_best_hier_similarities(classes, distances, max_dist):
    """
    For each class, create the optimal set of retrievals (used to calculate hierarchical precision @k)
    """
    distance_matrix = np.zeros([len(classes), len(classes)])
    best_hier_similarities = np.zeros([len(classes), len(classes)])

    for i in range(len(classes)):
        for j in range(len(classes)):
            distance_matrix[i, j] = distances[(classes[i], classes[j])]

    for i in range(len(classes)):
        best_hier_similarities[i, :] = 1 - np.sort(distance_matrix[i, :]) / max_dist

    return best_hier_similarities

'''
def _generate_summary(
        loss_accum,
        flat_accuracy_accums,
        hdist_accums,
        hdist_top_accums,
        hdist_mistakes_accums,
        hprecision_accums,
        hmAP_accums,
        num_logged,
        norm_mistakes_accum,
        loss_id,
        dist_id,
):
    """
    Generate dictionary with epoch's summary
    """
    summary = dict()
    summary[loss_id] = loss_accum / num_logged
    # -------------------------------------------------------------------------------------------------
    summary.update({accuracy_ids[i]: flat_accuracy_accums[i] / num_logged for i in range(len(topK_to_consider))})
    summary.update({dist_id + dist_avg_ids[i]: hdist_accums[i] / num_logged for i in range(len(topK_to_consider))})
    summary.update({dist_id + dist_top_ids[i]: hdist_top_accums[i] / num_logged for i in range(len(topK_to_consider))})
    summary.update(
        {dist_id + dist_avg_mistakes_ids[i]: hdist_mistakes_accums[i] / (norm_mistakes_accum * topK_to_consider[i]) for i in range(len(topK_to_consider))}
    )
    summary.update({dist_id + hprec_ids[i]: hprecision_accums[i] / num_logged for i in range(len(topK_to_consider))})
    summary.update({dist_id + hmAP_ids[i]: hmAP_accums[i] / num_logged for i in range(len(topK_to_consider))})
    return summary


def _update_tb_from_summary(summary, writer, steps, loss_id, dist_id):
    """
    Update tensorboard from the summary for the epoch
    """
    writer.add_scalar(loss_id, summary[loss_id], steps)

    for i in range(len(topK_to_consider)):
        writer.add_scalar(accuracy_ids[i], summary[accuracy_ids[i]] * 100, steps)
        writer.add_scalar(dist_id + dist_avg_ids[i], summary[dist_id + dist_avg_ids[i]], steps)
        writer.add_scalar(dist_id + dist_top_ids[i], summary[dist_id + dist_top_ids[i]], steps)
        writer.add_scalar(dist_id + dist_avg_mistakes_ids[i], summary[dist_id + dist_avg_mistakes_ids[i]], steps)
        writer.add_scalar(dist_id + hprec_ids[i], summary[dist_id + hprec_ids[i]] * 100, steps)
        writer.add_scalar(dist_id + hmAP_ids[i], summary[dist_id + hmAP_ids[i]] * 100, steps)
'''
