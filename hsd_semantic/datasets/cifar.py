import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import ConcatDataset
from .sampler import BatchSchedulerSampler
from hsd_semantic.config import config

data_path = os.path.join(config.DATASET.ROOT, 'cifar')

def get_transform(train=True):
    if train:
        transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    else:
        transform = transforms.Compose([
                transforms.Resize(32),transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    return transform

def get_CifarDataLoader():
    transform_train = get_transform(train=True)
    transform_test = get_transform(train=False)
    if config.DATASET.NAME == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)
    else:
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
        
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=config.SOLVER.BATCH_SIZE, 
                                              shuffle=True, 
                                              num_workers=config.SOLVER.WORKERS)
    testloader = torch.utils.data.DataLoader(testset, 
                                             batch_size=config.SOLVER.BATCH_SIZE,
                                             shuffle=False, 
                                             num_workers=config.SOLVER.WORKERS) 
    return trainloader, testloader


def subnetwise_Dataset(dataset, class_clusters):
    num_clusters = len(class_clusters)
    print(class_clusters)
    bins = {i: [] for i in range(num_clusters)}
    for i, y in enumerate(dataset.targets):
        for cluster in range(num_clusters):
            if y in class_clusters[cluster]:
                bins[cluster].append(i)
                break
    cluster_datasets = [torch.utils.data.Subset(dataset, indices) for k, indices in bins.items()]
    return cluster_datasets 
    

def subnetwise_DataLoader(class_clusters):
    transform_train = get_transform(train=True)
    transform_test = get_transform(train=False)
    if config.DATASET.NAME == 'cifar100':
        trainset_orig = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
        testset_orig = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)
    else:
        trainset_orig = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
        testset_orig = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)

    class_clusters = [tuple(clust) for clust in class_clusters]
    trainsets = subnetwise_Dataset(trainset_orig, class_clusters)
    testsets = subnetwise_Dataset(testset_orig, class_clusters)
    trainset = ConcatDataset(trainsets)
    testset = ConcatDataset(testsets)

    trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                         sampler=BatchSchedulerSampler(dataset=trainset,
                                                                       batch_size=config.SOLVER.BATCH_SIZE),
                                         batch_size=config.SOLVER.BATCH_SIZE,
                                         shuffle=False) #can't set shuffle True when working with sampler
    testloader = torch.utils.data.DataLoader(dataset=testset,
                                         sampler=BatchSchedulerSampler(dataset=testset,
                                                                       batch_size=config.SOLVER.BATCH_SIZE),
                                         batch_size=config.SOLVER.BATCH_SIZE,
                                         shuffle=False)
    return trainloader, testloader 
