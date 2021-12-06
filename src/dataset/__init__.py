import os

import torch
import torchvision
from dotmap import DotMap
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # suppress deprecation warning coming from torchvision
from torchvision import transforms as T

from src.dataset.celeba import CelebAGenerator
from src.dataset.transforms import transformer
from src.utils import config
from src.utils.config import get_config


def get_root(dataset_type):
    conf = get_config('config/default.env')
    return conf.dataset_root.__dict__[dataset_type]


def get_n_classes_and_channels(str_dataset):
    datasets = get_datasets(str_dataset)
    n_channels = int(datasets['train'][0][0].shape[0])
    return len(datasets.train.classes), n_channels


def get_data_loaders(str_dataset, seed=None, **kwargs):
    datasets = get_datasets(str_dataset)

    if seed is not None:
        torch.manual_seed(seed)

    common_settings = dict(batch_size=32, num_workers=4, drop_last=True)
    common_settings.update(**kwargs)

    if config.debug:
        batch_size = kwargs.get('batch_size', 50)
        datasets.train = data_utils.Subset(datasets.train, torch.arange(3 * batch_size + 1))
        datasets.val = data_utils.Subset(datasets.val, torch.arange(3 * batch_size + 1))

    train = DataLoader(datasets.train, shuffle=True, **common_settings)
    val = DataLoader(datasets.val, **common_settings)

    data_loaders = DotMap({'train': train, 'val': val})
    return data_loaders


def get_datasets(str_dataset):
    str_dataset = str_dataset.lower()

    if str_dataset == 'fashion':
        trans = transformer['mnist']
        return create_pytorch_datasets(torchvision.datasets.FashionMNIST, trans)
    elif str_dataset == 'mnist':
        trans = transformer['mnist']
        return create_pytorch_datasets(torchvision.datasets.MNIST, trans)
    elif str_dataset == 'cifar10':
        trans = transformer['cifar10']
        return create_pytorch_datasets(torchvision.datasets.CIFAR10, trans)
    elif str_dataset == 'cifar100':
        trans = transformer['cifar100']
        return create_pytorch_datasets(torchvision.datasets.CIFAR100, trans)
    elif str_dataset in ['celeba', 'celebagenerator']:
        trans = transformer['celeba']
        return create_celebalucid_datasets('celeba', trans)
    else:
        raise ValueError('Dataset {} is unknown.'.format(str_dataset))


def create_pytorch_datasets(dataset_func, trans):
    root = get_root('pytorch')
    train_trans = T.Compose(trans[0])
    val_trans = T.Compose(trans[1])

    try:
        train = dataset_func(root, train=True, transform=train_trans)
        val = dataset_func(root, train=False, transform=val_trans)
    except:
        train = dataset_func(root, train=True, download=True, transform=train_trans)
        val = dataset_func(root, train=False, download=True, transform=val_trans)

    return DotMap({'train': train, 'val': val})


def create_celebalucid_datasets(data_name, trans):
    root = get_root(data_name)

    csv = {
        'train': os.path.join(root, 'train.csv'),
        'val': os.path.join(root, 'test.csv')
    }

    trans = {
        'train': T.Compose(trans[0]),
        'val': T.Compose(trans[1])
    }

    train = CelebAGenerator(csv['train'], trans['train'])
    val = CelebAGenerator(csv['val'], trans['val'])
    return DotMap({'train': train, 'val': val})
