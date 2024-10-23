# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os, lmdb, pickle, six
from PIL import Image
import torch
from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform


class ImageFolderLMDB(torch.utils.data.Dataset):
    def __init__(self, db_path, transform=None):
        # 获取数据路径
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # pickle.loads()：反序列化从数据库中读取的字节数据，恢复成原来的 Python 对象。
            # 获取数据集的长度
            self.length = pickle.loads(txn.get(b'__len__'))
            # 获取数据集的key
            self.keys = pickle.loads(txn.get(b'__keys__'))
        self.transform = transform

# 用于根据索引从 LMDB 数据库中加载图像和标签数据，
# 并将图像转化为 RGB 模式，应用必要的预处理操作后返回
    def __getitem__(self, idx):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[idx])
        unpacked = pickle.loads(byteflow)

        # 加载图片
        imgbuf = unpacked[0]
        buf = six.BytesIO() 
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # 加载标签
        label = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        return img, label
    
# 返回数据集中样本的总数，用于指示数据集的长度
    def __len__(self):
        return self.length

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'IMNET_LMDB':
        print("reading from datapath", args.data_path)
        path = os.path.join(args.data_path, 'train.lmdb' if is_train else 'val.lmdb')
        dataset = ImageFolderLMDB(path, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32    # 识别图片大小，来判断是否要resize_image
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std  # 参数的均值和方差
    # 是否使用imagenet的均值和方差
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # 这应该始终发送到transforms_imagenet_train
        # 定义transform的一些参数
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        # 如果图片大小小于32，那就进行随机裁剪
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

# 测试阶段的处理
    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(args.input_size))
    # 图像转换为tensor
    t.append(transforms.ToTensor())
    # 归一化
    t.append(transforms.Normalize(mean, std))
    # 将所有的 transform 操作组合成一个顺序执行的 pipeline 并返回
    return transforms.Compose(t)
