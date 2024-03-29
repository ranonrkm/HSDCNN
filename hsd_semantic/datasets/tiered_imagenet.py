import os
import torch
import torchvision
import torchvision.transforms as transforms
from hsd_semantic.config import config

data_path = os.path.join(config.DATASET.ROOT, 'ImageNet', 'tiered_imagenet')

mean_ilsvrc12 = [0.485, 0.456, 0.406]
std_ilsvrc12 = [0.229, 0.224, 0.225]
mean_inat19 = [0.454, 0.474, 0.367]
std_inat19 = [0.237, 0.230, 0.249]

normalize_tfs_ilsvrc12 = transforms.Normalize(mean=mean_ilsvrc12, std=std_ilsvrc12)
normalize_tfs_inat19 = transforms.Normalize(mean=mean_inat19, std=std_inat19)
normalize_tfs_dict = {
    "tiered-imagenet-84": normalize_tfs_ilsvrc12,
    "tiered-imagenet-224": normalize_tfs_ilsvrc12,
    "ilsvrc12": normalize_tfs_ilsvrc12,
    "inaturalist19-84": normalize_tfs_inat19,
    "inaturalist19-224": normalize_tfs_inat19,
}


def train_transforms(img_resolution, dataset, augment=True, normalize=True):
    if augment and normalize:
        return transforms.Compose(
            [
                # extract random crops and resize to img_resolution
                transforms.RandomResizedCrop(img_resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_tfs_dict[dataset],
            ]
        )
    elif not augment and normalize:
        return transforms.Compose([transforms.ToTensor(), normalize_tfs_dict[dataset]])
    elif augment and not normalize:
        return transforms.Compose([transforms.RandomResizedCrop(img_resolution), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    else:
        return transforms.Compose([transforms.ToTensor()])


def val_transforms(dataset, normalize=True, resize=None, crop=None):
    trsfs = []

    if resize:
        trsfs.append(transforms.Resize(resize))

    if crop:
        trsfs.append(transforms.CenterCrop(crop))

    if normalize:
        trsfs.extend([transforms.ToTensor(), normalize_tfs_dict[dataset]])
    else:
        trsfs.append([*transforms.ToTensor()])

    return transforms.Compose(trsfs)

def get_TImageNetLoader():
    dataset_name = config.DATASET.NAME + '-' + str(config.DATASET.RES)
    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val')
    train_dataset = torchvision.datasets.ImageFolder(train_dir, train_transforms(config.DATASET.RES, dataset_name, augment=config.DATASET.AUGMENT, normalize=True))
    val_dataset = torchvision.datasets.ImageFolder(val_dir, val_transforms(dataset_name, normalize=True))

    train_loader = torch.utils.data.DataLoader(
                            train_dataset, 
                            batch_size=config.SOLVER.BATCH_SIZE, 
                            shuffle=True, num_workers=config.SOLVER.WORKERS, 
                            pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
                            val_dataset, 
                            batch_size=config.SOLVER.BATCH_SIZE, 
                            shuffle=False, num_workers=config.SOLVER.WORKERS, 
                            pin_memory=True, drop_last=True)


    return train_loader, val_loader


