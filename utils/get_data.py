# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torchvision.transforms as transforms
import torchvision.datasets 
from .datasets import *

from .cifar_label import *
def get_dataset(dataset_name, data_dir, split, rand_fraction=None,clean=False, transform=None, imsize=None, bucket='pytorch-data', **kwargs):
  if dataset_name in [ 'cifar10', 'cifar100','cifar10T','cifar100T', 'food101','imagenette']:
    dataset = globals()[f'get_{dataset_name}'](dataset_name, data_dir, split, transform=imsize, imsize=imsize, bucket=bucket, **kwargs)    
  elif dataset_name in ['cifar100N']:
    dataset = globals()[f'get_{dataset_name}'](dataset_name, data_dir, split,rand_fraction= rand_fraction,transform=imsize, imsize=imsize, bucket=bucket,**kwargs)
  item = dataset.__getitem__(0)[0]
  print (item.size(0))
  dataset.nchannels = item.size(0)
  dataset.imsize = item.size(1)
  return dataset


def get_aug(split, imsize=None, aug='large'):
  if aug == 'large':
    imsize = imsize if imsize is not None else 224
    if split == 'train':
      return [transforms.RandomResizedCrop(imsize, scale=(0.2, 1.0)),transforms.RandomHorizontalFlip()]
      #return [transforms.Resize(round(imsize * 1.143)), transforms.CenterCrop(imsize)]
    else:
      return [transforms.Resize(round(imsize * 1.143)), transforms.CenterCrop(imsize)]
  else:
    imsize = imsize if imsize is not None else 32
    if split == 'train':
        train_transform = []
        print("Using CIFAR-10/100 style augmentation")
      #return [transforms.RandomCrop(imsize, padding=round(imsize / 8))]
        train_transform.append(transforms.RandomCrop(32, padding=4))
        train_transform.append(transforms.RandomHorizontalFlip())
        return train_transform
    else:
      return [transforms.Resize(imsize), transforms.CenterCrop(imsize)]


def get_transform(split, normalize=None, transform=None, imsize=None, aug='large'):
  if transform is None:
    if normalize is None:
        if aug == 'large':
          normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
          normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  
    transform = transforms.Compose(get_aug(split, imsize=imsize, aug=aug)
                                   + [transforms.ToTensor(), normalize])
  return transform

def get_imagenette(dataset_name, data_dir, split, transform=None, imsize=None, bucket='pytorch-data', **kwargs):
  d_split = split
  t_split = split
  if split == 'holdout':
    d_split = 'train'
    t_split = 'val'
    
  transform = get_transform(t_split, transform=transform, imsize=imsize, aug='large')
  return Imagenette(data_dir, split=d_split, transform=transform, download=False, **kwargs)

def get_food101(dataset_name, data_dir, split, transform=None, imsize=None, bucket='pytorch-data', **kwargs):
  d_split = split
  t_split = split
  if split == 'holdout':
    d_split = 'train'
    t_split = 'val'

  transform = get_transform(t_split, transform=transform, imsize=imsize, aug='large')
  return torchvision.datasets.Food101(data_dir, split=d_split, transform=transform, download=True, **kwargs)

def get_cifar10(dataset_name, data_dir, split, transform=None, imsize=None, bucket='pytorch-data', **kwargs):
  d_split = split
  t_split = split
  if split == 'holdout':
    d_split = 'train'
    t_split = 'val'
  transform = get_transform(split, transform=transform, imsize=imsize, aug='small')
  return torchvision.datasets.CIFAR10(data_dir, train=(split=='train'), transform=transform, download=True, **kwargs)

def get_cifar10T(dataset_name, data_dir, split, transform=None, imsize=None, bucket='pytorch-data', **kwargs):
  d_split = split
  t_split = split
  if split == 'holdout':
    d_split = 'train'
    t_split = 'val'

  transform = get_transform(t_split, transform=transform, imsize=imsize, aug='small')
  # if t_split == 'train' :
  #   from timm.data import create_transform
  #   transform = create_transform(
  #     input_size=32,
  #     is_training=True,
  #     color_jitter=0.4,
  #     auto_augment='rand-m9-mstd0.5-inc1',
  #     interpolation='bicubic',
  #     re_prob=0.25,
  #     re_mode='pixel',
  #     re_count=1,
  #     mean=[0.4914, 0.4822, 0.4465],
  #     std=[0.2023, 0.1994, 0.2010],
  # )

  return CIFAR10T(data_dir, train=(d_split=='train'), transform=transform, download=True, **kwargs)

def get_cifar100T(dataset_name, data_dir, split, transform=None, imsize=None, bucket='pytorch-data', **kwargs):
  d_split = split
  t_split = split
  if split == 'holdout':
    d_split = 'train'
    t_split = 'val'
  transform = get_transform(t_split, transform=transform, imsize=imsize, aug='small')
  return CIFAR100T(data_dir, train=(d_split=='train'), transform=transform, download=True, **kwargs)

def get_cifar100(dataset_name, data_dir, split, transform=None, imsize=None, bucket='pytorch-data', **kwargs):
  transform = get_transform(split, transform=transform, imsize=imsize, aug='small')
  return torchvision.datasets.CIFAR100(data_dir, train=(split=='train'), transform=transform, download=True, **kwargs)

def get_cifar100N(dataset_name, data_dir, split, rand_fraction=None,transform=None, imsize=None, bucket='pytorch-data', **kwargs):
  transform = get_transform(split, transform=transform, imsize=imsize, aug='small')
  if split=='train':
    return CIFAR100N(root=data_dir, train=(split=='train'), transform=transform, download=True, rand_fraction=rand_fraction)
  else:
    return torchvision.datasets.CIFAR100(data_dir, train=(split=='train'), transform=transform, download=True, **kwargs)        

