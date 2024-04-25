import logging
import os
import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

logger = logging.getLogger(__name__)

def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size)),
        transforms.ToTensor(),
        #transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        #transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(args.data_dir, 'train')  # Update with your ImageNet train data directory
    val_dir = os.path.join(args.data_dir, 'test')  # Update with your ImageNet validation data directory
    # train_dir = os.path.join(args.data_dir, './train')  # Update with your ImageNet train data directory
    # val_dir = os.path.join(args.data_dir, './val')  # Update with your ImageNet validation data directory
    trainset = datasets.ImageFolder(train_dir, transform=transform_train)
    valset = datasets.ImageFolder(val_dir, transform=transform_val) if args.local_rank in [-1, 0] else None

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    val_sampler = SequentialSampler(valset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(valset,
                             sampler=val_sampler,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             pin_memory=True) if valset is not None else None

    return trainset, valset, train_loader, val_loader

