import os
import torch
import torchvision
import torchvision.transforms as transforms
from hsd_semantic.config import config

data_path = os.path.join(config.DATASET.ROOT, 'cifar')

def get_CifarDataLoader():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    transform_test = transforms.Compose([
        transforms.Resize(32),transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

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


def get_classwise_DataLoader():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    
    if config.DATASET.NAME == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
    else:
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)

    classwise_trainset = {}
    for class_id in range(config.DATASET.NUM_CLASSES):
        class_images = []
        class_labels = []
        for (images, labels) in trainloader:
            class_samples = torch.eq(labels, class_id)
            class_labels.extend([labels[i] for i in range(len(labels)) if class_samples[i].item() == 1])
            class_images.extend([images[i] for i in range(len(labels)) if class_samples[i].item() == 1])
        class_images = torch.stack(class_images, dim=0)
        class_labels = torch.stack(class_labels, dim=0)
        classwise_trainset[class_id] = torch.utils.data.TensorDataset(class_images, class_labels)

    return classwise_trainset
