# coding: utf-8
import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as data
import torchvision
from torchvision import transforms, models
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from skimage import io, transform
import numpy as np
import os
from PIL import Image
import random
from swin_transformer import SwinTransformer
from itertools import groupby
import math

# from thop import profile
# import sys
# from thop import clever_format
# from torchstat import stat


def parse_args():
    """
    Parse input arguments
    Returns
    -------
    args : object
        Parsed args
    """
    h = {
        "program": "Simple Baselines training",
        "train_folder": "Path to training data folder.",
        "batch_size": "Number of images to load per batch. Set according to your PC GPU memory available. If you get "
                      "out-of-memory errors, lower the value. defaults to 64",
        "epochs": "How many epochs to train for. Once every training image has been shown to the CNN once, an epoch "
                  "has passed. Defaults to 15",
        "test_folder": "Path to test data folder",
        "num_workers": "Number of workers to load in batches of data. Change according to GPU usage",
        "test_only": "Set to true if you want to test a loaded model. Make sure to pass in model path",
        "model_path": "Path to your model",
        "learning_rate": "The learning rate of your model. Tune it if it's overfitting or not learning enough"}
    parser = argparse.ArgumentParser(description=h['program'], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_folder', help=h["train_folder"], type=str, default='../train_images')
    # parser.add_argument('--train_folder', help=h["train_folder"], type=str, default='./Track1/train_images')
    # parser.add_argument('--batch_size', help=h['batch_size'], type=int, default=32)
    parser.add_argument('--batch_size', help=h['batch_size'], type=int, default=16)
    # parser.add_argument('--epochs', help=h["epochs"], type=int, default=15)
    parser.add_argument('--epochs', help=h["epochs"], type=int, default=500)
    parser.add_argument('--test_folder', help=h["test_folder"], type=str)
    parser.add_argument('--num_workers', help=h["num_workers"], type=int, default=2)
    parser.add_argument('--test_only', help=h["test_only"], type=bool, default=False)
    parser.add_argument('--model_path', help=h["num_workers"], type=str),
    parser.add_argument('--learning_rate', help=h["learning_rate"], type=float, default=0.003)

    args = parser.parse_args()

    return args


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


class MyData(Dataset):  # 继承Dataset
    def __init__(self, root_dir, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.transform = transform  # 变换

        self.all_files = []
        for dir_name in os.listdir(self.root_dir):
            for file in os.listdir(self.root_dir + '/' + dir_name):
                # print(os.path.join(root, file))
                if 'EO' not in file:
                    self.all_files.append(self.root_dir + '/' + dir_name + '/' + file)
        # print(self.all_files)
        # for item in self.all_files:
        #     if 'EO' in item:
        #         self.all_files.remove(item)
        # print(self.all_files)
        random.seed(0)
        np.random.seed(0)
        random.shuffle(self.all_files)
        self.all_files = self.all_files[:int(0.03 * (len(self.all_files)))]

    def __len__(self):  # 返回整个数据集的大小
        return len(self.all_files)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        # img_path = os.path.join(self.root_dir, image_index)  # 获取索引为index的图片的路径名
        img_path = self.all_files[index]
        # img = io.imread(img_path)  # 读取该图片
        img = Image.open(img_path)  # 读取该图片
        img = img.convert('RGB')
        label = int(img_path.split('/')[-2])
        # print(label)
        # sample = {'image': img, 'label': label}  # 根据图片和标签创建字典

        if self.transform:
            # sample = self.transform(sample)  # 对样本进行变换
            img = self.transform(img)  # 对样本进行变换
        # return sample  # 返回该样本
        return img, label  # 返回该样本


class MyData2(Dataset):  # 继承Dataset
    def __init__(self, root_dir, flag, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.transform = transform  # 变换

        self.all_files_train = []
        self.all_files_test = []
        for dir_name in os.listdir(self.root_dir):
            dir_files = []
            for file in os.listdir(self.root_dir + '/' + dir_name):
                # print(os.path.join(root, file))
                if 'EO' not in file:
                    # self.all_files.append(self.root_dir + '/' + dir_name + '/' + file)
                    dir_files.append(self.root_dir + '/' + dir_name + '/' + file)
            random.seed(0)
            np.random.seed(0)
            random.shuffle(dir_files)
            test_files = dir_files[:100]
            train_files = dir_files[100:]
            # train_files = dir_files[100:500]

            random.seed(None)
            np.random.seed(None)
            random.shuffle(train_files)
            train_files = train_files[:1000]

            self.all_files_test += test_files
            self.all_files_train += train_files

        if flag == 0:   # test
            self.all_files = self.all_files_test
        elif flag == 1:   # train
            self.all_files = self.all_files_train

            # random.seed(None)
            # np.random.seed(None)
            # random.shuffle(self.all_files)
            # # self.all_files = self.all_files[:int(0.1 * (len(self.all_files)))]
            # self.all_files = self.all_files[:int(0.1 * (len(self.all_files)))]

        # print(self.all_files)
        # for item in self.all_files:
        #     if 'EO' in item:
        #         self.all_files.remove(item)
        # print(self.all_files)

        self.targets = self.cal_concat_targets()

    def __len__(self):  # 返回整个数据集的大小
        return len(self.all_files)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        # img_path = os.path.join(self.root_dir, image_index)  # 获取索引为index的图片的路径名
        img_path = self.all_files[index]
        # img = io.imread(img_path)  # 读取该图片
        img = Image.open(img_path)  # 读取该图片
        img = img.convert('RGB')
        label = int(img_path.split('/')[-2])
        # print(label)
        # sample = {'image': img, 'label': label}  # 根据图片和标签创建字典

        if self.transform:
            # sample = self.transform(sample)  # 对样本进行变换
            img = self.transform(img)  # 对样本进行变换
        # return sample  # 返回该样本
        return img, label  # 返回该样本

    def cal_concat_targets(self):
        concat_targets = []
        for i in range(len(self.all_files)):
            label = int(self.all_files[i].split('/')[-2])
            concat_targets.append(label)
        # print(concat_targets)
        return concat_targets


class MyData3(Dataset):  # 继承Dataset
    def __init__(self, root_dir, flag, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.transform = transform  # 变换

        self.all_files_train = []
        self.all_files_test = []
        for dir_name in os.listdir(self.root_dir):
            dir_files = []
            for file in os.listdir(self.root_dir + '/' + dir_name):
                # print(os.path.join(root, file))
                if 'EO' not in file:
                    # self.all_files.append(self.root_dir + '/' + dir_name + '/' + file)
                    dir_files.append(self.root_dir + '/' + dir_name + '/' + file)
            random.seed(0)
            np.random.seed(0)
            random.shuffle(dir_files)
            test_files = dir_files[:100]
            train_files = dir_files[100:]
            # train_files = dir_files[100:500]

            # random.seed(None)
            # np.random.seed(None)
            # random.shuffle(train_files)
            # train_files = train_files[:1000]

            self.all_files_test += test_files
            self.all_files_train += train_files

        if flag == 0:   # test
            self.all_files = self.all_files_test
        elif flag == 1:   # train
            self.all_files = self.all_files_train

        self.flag = flag

            # random.seed(None)
            # np.random.seed(None)
            # random.shuffle(self.all_files)
            # # self.all_files = self.all_files[:int(0.1 * (len(self.all_files)))]
            # self.all_files = self.all_files[:int(0.1 * (len(self.all_files)))]

        # print(self.all_files)
        # for item in self.all_files:
        #     if 'EO' in item:
        #         self.all_files.remove(item)
        # print(self.all_files)

        self.targets = self.cal_concat_targets()
        # print(self.targets)

        if self.flag == 1:
            self.c0 = np.where(np.array(self.targets)==0)[0].tolist()
            self.c1 = np.where(np.array(self.targets)==1)[0].tolist()
            self.c2 = np.where(np.array(self.targets)==2)[0].tolist()
            self.c3 = np.where(np.array(self.targets)==3)[0].tolist()
            self.c4 = np.where(np.array(self.targets)==4)[0].tolist()
            self.c5 = np.where(np.array(self.targets)==5)[0].tolist()
            self.c6 = np.where(np.array(self.targets)==6)[0].tolist()
            self.c7 = np.where(np.array(self.targets)==7)[0].tolist()
            self.c8 = np.where(np.array(self.targets)==8)[0].tolist()
            self.c9 = np.where(np.array(self.targets)==9)[0].tolist()
            # print(len(self.c0))
        elif self.flag == 0:
            pass

    def __len__(self):  # 返回整个数据集的大小
        # return len(self.all_files)
        if self.flag == 1:
            return 500 * 10
        elif self.flag == 0:
            return len(self.all_files)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        # img_path = os.path.join(self.root_dir, image_index)  # 获取索引为index的图片的路径名
        # print(index)

        if self.flag == 1:

            random.seed(None)
            np.random.seed(None)
            c0 = random.sample(self.c0, 500)
            c1 = random.sample(self.c1, 500)
            c2 = random.sample(self.c2, 500)
            c3 = random.sample(self.c3, 500)
            c4 = random.sample(self.c4, 500)
            c5 = random.sample(self.c5, 500)
            c6 = random.sample(self.c6, 500)
            c7 = random.sample(self.c7, 500)
            c8 = random.sample(self.c8, 500)
            c9 = random.sample(self.c9, 500)

            updating_train_list = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9
            updating_train_list = np.array(self.all_files)[updating_train_list].tolist()

            # img_path = self.all_files[index]
            img_path = updating_train_list[index]

        elif self.flag == 0:
            img_path = self.all_files[index]

        # img = io.imread(img_path)  # 读取该图片
        img = Image.open(img_path)  # 读取该图片
        img = img.convert('RGB')
        label = int(img_path.split('/')[-2])

        if self.transform:
            # sample = self.transform(sample)  # 对样本进行变换
            img = self.transform(img)  # 对样本进行变换
        # return sample  # 返回该样本
        return img, label  # 返回该样本

    def cal_concat_targets(self):
        concat_targets = []
        for i in range(len(self.all_files)):
            label = int(self.all_files[i].split('/')[-2])
            concat_targets.append(label)
        # print(concat_targets)
        return concat_targets


class MyData4(Dataset):  # 继承Dataset
    def __init__(self, root_dir, flag, transform_SAR=None, transform_EO=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.transform_SAR = transform_SAR  # 变换
        self.transform_EO = transform_EO  # 变换

        self.all_files_train = []
        self.all_files_test = []
        for dir_name in os.listdir(self.root_dir):
            dir_files = []
            for file in os.listdir(self.root_dir + '/' + dir_name):
                # print(os.path.join(root, file))
                if 'EO' not in file:
                    # self.all_files.append(self.root_dir + '/' + dir_name + '/' + file)
                    dir_files.append(self.root_dir + '/' + dir_name + '/' + file)   # append the SAR image
            random.seed(0)
            np.random.seed(0)
            random.shuffle(dir_files)
            test_files = dir_files[:100]
            train_files = dir_files[100:]
            # train_files = dir_files[100:500]

            # random.seed(None)
            # np.random.seed(None)
            # random.shuffle(train_files)
            # train_files = train_files[:1000]

            self.all_files_test += test_files
            self.all_files_train += train_files

        if flag == 0:   # test
            self.all_files = self.all_files_test
            # self.all_files = self.all_files_train
        elif flag == 1:   # train
            self.all_files = self.all_files_train

        self.flag = flag

            # random.seed(None)
            # np.random.seed(None)
            # random.shuffle(self.all_files)
            # # self.all_files = self.all_files[:int(0.1 * (len(self.all_files)))]
            # self.all_files = self.all_files[:int(0.1 * (len(self.all_files)))]

        # print(self.all_files)
        # for item in self.all_files:
        #     if 'EO' in item:
        #         self.all_files.remove(item)
        # print(self.all_files)

        self.targets = self.cal_concat_targets()
        # print(self.targets)

        if self.flag == 1:
            self.c0 = np.where(np.array(self.targets)==0)[0].tolist()
            self.c1 = np.where(np.array(self.targets)==1)[0].tolist()
            self.c2 = np.where(np.array(self.targets)==2)[0].tolist()
            self.c3 = np.where(np.array(self.targets)==3)[0].tolist()
            self.c4 = np.where(np.array(self.targets)==4)[0].tolist()
            self.c5 = np.where(np.array(self.targets)==5)[0].tolist()
            self.c6 = np.where(np.array(self.targets)==6)[0].tolist()
            self.c7 = np.where(np.array(self.targets)==7)[0].tolist()
            self.c8 = np.where(np.array(self.targets)==8)[0].tolist()
            self.c9 = np.where(np.array(self.targets)==9)[0].tolist()
            # print(len(self.c0))
        elif self.flag == 0:
            pass

    def __len__(self):  # 返回整个数据集的大小
        # return len(self.all_files)
        if self.flag == 1:
            return 500 * 10
        elif self.flag == 0:
            return len(self.all_files)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        # img_path = os.path.join(self.root_dir, image_index)  # 获取索引为index的图片的路径名
        # print(index)

        if self.flag == 1:

            random.seed(None)
            np.random.seed(None)
            c0 = random.sample(self.c0, 500)
            c1 = random.sample(self.c1, 500)
            c2 = random.sample(self.c2, 500)
            c3 = random.sample(self.c3, 500)
            c4 = random.sample(self.c4, 500)
            c5 = random.sample(self.c5, 500)
            c6 = random.sample(self.c6, 500)
            c7 = random.sample(self.c7, 500)
            c8 = random.sample(self.c8, 500)
            c9 = random.sample(self.c9, 500)

            updating_train_list = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9
            updating_train_list = np.array(self.all_files)[updating_train_list].tolist()

            # img_path = self.all_files[index]
            img_path = updating_train_list[index]

        elif self.flag == 0:
            img_path = self.all_files[index]

        # img = io.imread(img_path)  # 读取该图片
        img_SAR = Image.open(img_path)  # 读取该图片
        img_path_EO = img_path.replace('SAR', 'EO')
        img_EO = Image.open(img_path_EO)
        img_SAR = img_SAR.convert('RGB')
        img_EO = img_EO.convert('RGB')
        label = int(img_path.split('/')[-2])

        if self.transform_SAR:
            # sample = self.transform(sample)  # 对样本进行变换
            img_SAR = self.transform_SAR(img_SAR)  # 对样本进行变换
        if self.transform_EO:
            # sample = self.transform(sample)  # 对样本进行变换
            img_EO = self.transform_EO(img_EO)  # 对样本进行变换
        # return sample  # 返回该样本
        # return img, label  # 返回该样本
        return img_SAR, img_EO, label  # 返回a pair样本and their label

    def cal_concat_targets(self):
        concat_targets = []
        for i in range(len(self.all_files)):
            label = int(self.all_files[i].split('/')[-2])
            concat_targets.append(label)
        # print(concat_targets)
        return concat_targets


def load_train_data(train_data_path, batch_size):
    # Convert images to tensors, normalize, and resize them
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform)
    train_data = MyData(train_data_path, transform=transform)  # 初始化类，设置数据集所在路径以及变换

    train_data_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

    return train_data_loader


def load_train_myval_data(train_data_path, batch_size):
    # Convert images to tensors, normalize, and resize them
    # transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.3984, 0.3984, 0.3984), (0.1328, 0.1328, 0.1328))])
    transform_train = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.ToTensor(), transforms.Normalize((0.3984, 0.3984, 0.3984), (0.1328, 0.1328, 0.1328))])
    # transform_train = transforms.Compose([transforms.Resize(80), transforms.RandomCrop(64), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transform_test = transforms.Compose([transforms.Resize(64), transforms.ToTensor(), transforms.Normalize((0.3984, 0.3984, 0.3984), (0.1328, 0.1328, 0.1328))])
    transform_test = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.ToTensor(), transforms.Normalize((0.3984, 0.3984, 0.3984), (0.1328, 0.1328, 0.1328))])
    train_data_raw_train = MyData3(train_data_path, flag=1, transform=transform_train)  # 初始化类，设置数据集所在路径以及变换
    train_data_raw_test = MyData3(train_data_path, flag=0, transform=transform_test)  # 初始化类，设置数据集所在路径以及变换

    train_data_loader = data.DataLoader(train_data_raw_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    # test_data_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    test_data_loader = data.DataLoader(train_data_raw_test, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    return train_data_loader, test_data_loader


def get_mean_std_value(loader):
    channels_sum_SAR, channel_squared_sum_SAR, channels_sum_EO, channel_squared_sum_EO, num_batches = 0,0,0,0,0

    for data_SAR, data_EO, target in loader:
        channels_sum_SAR += torch.mean(data_SAR, dim=[0,2,3])#shape [n_samples(batch),channels,height,width]
        #并不需要求channel的均值
        channel_squared_sum_SAR += torch.mean(data_SAR**2, dim=[0,2,3])#shape [n_samples(batch),channels,height,width]
        channels_sum_EO += torch.mean(data_EO, dim=[0,2,3])#shape [n_samples(batch),channels,height,width]
        #并不需要求channel的均值
        channel_squared_sum_EO += torch.mean(data_EO**2, dim=[0,2,3])#shape [n_samples(batch),channels,height,width]
        num_batches += 1

    # This lo calculate the summarized value of mean we need to divided it by num_batches

    mean_SAR = channels_sum_SAR/num_batches
    mean_EO = channels_sum_EO/num_batches
    #这里将标准差的公式变形了一下，让代码更方便写一点

    std_SAR = (channel_squared_sum_SAR/num_batches - mean_SAR**2)**0.5
    std_EO = (channel_squared_sum_EO/num_batches - mean_EO**2)**0.5
    # return mean, std
    return mean_SAR, std_SAR, mean_EO, std_EO


def load_train_myval_pairdata(train_data_path, batch_size):
    # Convert images to tensors, normalize, and resize them
    # transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.3984, 0.3984, 0.3984), (0.1328, 0.1328, 0.1328))])
    # transform_train = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.ToTensor(), transforms.Normalize((0.3984, 0.3984, 0.3984), (0.1328, 0.1328, 0.1328))])
    # transform_train_SAR = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.ToTensor(), transforms.Normalize((0.4103, 0.4103, 0.4103), (0.1280, 0.1280, 0.1280))])
    # transform_train_EO = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.ToTensor(), transforms.Normalize((0.2930, 0.2930, 0.2930), (0.1479, 0.1479, 0.1479))])
    transform_train_SAR = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(10, resample=False, expand=False, center=None), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_train_EO = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(10, resample=False, expand=False, center=None), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transform_train = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transform_train = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.ToTensor()])
    # transform_train = transforms.Compose([transforms.Resize(80), transforms.RandomCrop(64), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transform_test = transforms.Compose([transforms.Resize(64), transforms.ToTensor(), transforms.Normalize((0.3984, 0.3984, 0.3984), (0.1328, 0.1328, 0.1328))])
    # transform_test = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.ToTensor(), transforms.Normalize((0.3984, 0.3984, 0.3984), (0.1328, 0.1328, 0.1328))])
    # transform_test = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_test_SAR = transform_train_SAR
    transform_test_EO = transform_train_EO
    train_data_raw_train = MyData4(train_data_path, flag=1, transform_SAR=transform_train_SAR, transform_EO=transform_train_EO)  # 初始化类，设置数据集所在路径以及变换
    # train_data_raw_train = MyData4(train_data_path, flag=0, transform=transform_train)  # 初始化类，设置数据集所在路径以及变换
    train_data_raw_test = MyData4(train_data_path, flag=0, transform_SAR=transform_test_SAR, transform_EO=transform_test_EO)  # 初始化类，设置数据集所在路径以及变换

    train_data_loader = data.DataLoader(train_data_raw_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    # test_data_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    # mean_SAR, std_SAR, mean_EO, std_EO = get_mean_std_value(train_data_loader)
    # print(mean_SAR, std_SAR, mean_EO, std_EO)

    test_data_loader = data.DataLoader(train_data_raw_test, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    return train_data_loader, test_data_loader


def load_test_data(test_data_path, batch_size):
    transform = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=transform)
    test_data_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

    return test_data_loader


class Classifier(nn.Module):
    def __init__(self, num_class):
        super(Classifier, self).__init__()
        self.ln1 = nn.Linear(1024, 4096)
        self.ln2 = nn.Linear(4096, 4096)
        self.ln3 = nn.Linear(4096, num_class)
        # self.ln = nn.Linear(1024, num_class)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.ln1(x))
        x1 = self.relu(self.ln2(x1))
        x1 = self.ln3(x1)
        # x1 = self.ln(x)
        return x1


class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        print("Mish activation loaded...")
    def forward(self,x):
        x = x * (torch.tanh(nn.functional.softplus(x)))
        return x


class ATT_CLS(nn.Module):
    def __init__(self, num_class):
        super(ATT_CLS, self).__init__()
        self.ln1 = nn.Linear(1024, 128 * 3)
        self.ln2 = nn.Linear(1024, 128 * 3)
        self.ln3 = nn.Linear(256, 256)
        self.ln4 = nn.Linear(256, 256)
        self.ln5 = nn.Linear(256, num_class)
        # self.ln = nn.Linear(1024, num_class)
        self.relu = nn.ReLU()
        self.mish = Mish()

    def forward(self, x1, x2):
        f1 = self.ln1(x1)
        f1_q, f1_k, f1_v = f1[:, :128], f1[:, 128:128*2], f1[:, 128*2:128*3]
        f2 = self.ln2(x2)
        f2_q, f2_k, f2_v = f2[:, :128], f2[:, 128:128*2], f2[:, 128*2:128*3]
        attn1 = torch.mm(f2_q, f1_k.transpose(1, 0))
        attn2 = torch.mm(f1_q, f2_k.transpose(1, 0))
        attn1 = nn.functional.softmax(attn1 / math.sqrt(128), -1)
        attn2 = nn.functional.softmax(attn2 / math.sqrt(128), -1)
        output1 = torch.mm(attn1, f1_v)
        output2 = torch.mm(attn2, f2_v)
        output = torch.cat((output1, output2), -1)
        output = self.mish(self.ln3(output))
        output = self.mish(self.ln4(output))
        output = self.ln5(output)
        return output


def train():
    args = parse_args()
    # train_data = load_train_data(args.train_folder, args.batch_size)
    # train_data, test_data = load_train_myval_data(args.train_folder, args.batch_size)
    train_data, test_data = load_train_myval_pairdata(args.train_folder, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    F = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=10,
                        embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                        window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                        # window_size=14, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                        use_checkpoint=False
                        # use_checkpoint=True
                        )  # the feature dim is 1024

    F.apply(inplace_relu)
    F.load_state_dict(torch.load('swin_base_patch4_window7_224_22k.pth')['model'], strict=False)  # use the pretrained model trained on ImageNet
    F = F.cuda()
    C = Classifier(10)
    C.apply(inplace_relu)
    C = C.cuda()
    F.train()
    C.train()

    # input = torch.randn(1,3,224,224).cuda()
    # macs, params = profile(F, inputs=(input, ))
    #
    # print(F.flops)
    #
    # print(macs)
    # print('---------------------------------')
    # print(params)
    # macs, params = clever_format([macs, params], "%.3f")

    # input = torch.randn(1, 1024).cuda()
    # macs, params = profile(C, inputs=(input,))
    #
    # print(macs)
    # print('---------------------------------')
    # print(params)
    #
    # macs, params = clever_format([macs, params], "%.3f")

    # total_params = sum(p.numel() for p in F.parameters())
    # print(f'{total_params:,} total parameters.')
    # print(f'{total_params/(1024*1024):.2f}M total parameters.')
    # total_trainable_params = sum( p.numel() for p in F.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')
    # print(f'{total_trainable_params/(1024*1024):.2f}M training parameters.')
    #
    # total_params = sum(p.numel() for p in C.parameters())
    # print(f'{total_params:,} total parameters.')
    # print(f'{total_params/(1024*1024):.2f}M total parameters.')
    # total_trainable_params = sum( p.numel() for p in C.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')
    # print(f'{total_trainable_params/(1024*1024):.2f}M training parameters.')

    # stat(F.cpu(), (3, 224, 224))
    #
    # sys.exit(0)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer_F = optim.AdamW(F.parameters(), lr=0.00002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    # optimizer_AC = optim.AdamW(AC.parameters(), lr=0.00002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    # optimizer_AC = optim.SGD(AC.parameters(), lr=0.001, weight_decay=1e-4)
    # optimizer_F = optim.SGD(F.parameters(), lr=0.001, weight_decay=1e-4)
    optimizer_C = optim.SGD(C.parameters(), lr=0.001, weight_decay=1e-4)
    # model.to(device)

    best_acc = 0.

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        # with tqdm(train_data, unit="batch") as tepoch:
        #     for inputs, labels in tepoch:
        F.train()
        C.train()
        # for inputs, labels in train_data:
        # train_data, _ = load_train_myval_data(args.train_folder, args.batch_size)
        for inputs_SAR, inputs_EO, labels in tqdm(train_data):

            inputs_SAR, inputs_EO, labels = inputs_SAR.to(device), inputs_EO.to(device), labels.to(device)
            # if inputs.shape[1] == 1:
            #     inputs = inputs.repeat(1, 3, 1, 1)

            # zero the parameter gradients
            # optimizer.zero_grad()
            optimizer_F.zero_grad()
            optimizer_C.zero_grad()

            # forward + get predictions + backward + optimize
            # outputs_SAR = F(inputs_SAR)
            outputs_EO = F(inputs_EO)
            # outputs_SAR = outputs_SAR.view(outputs_SAR.shape[0], -1)
            outputs_EO = outputs_EO.view(outputs_EO.shape[0], -1)
            outputs = C(outputs_EO)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
            print(predictions)
            print(labels)
            correct = (predictions == labels).sum().item()
            accuracy = correct / inputs_SAR.shape[0]

            loss.backward()
            optimizer_F.step()
            optimizer_C.step()

            # tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
            print('loss {:.4f}  acc {:.4f}'.format(loss.item(), 100. * accuracy))

        F.eval()
        C.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            # all_pred_outputs = torch.zeros(10, 770, 10)
            all_pred_outputs = torch.zeros(10, 1000, 10)
            # all_pred_outputs = torch.zeros(1, 1000, 10)
            for iter in range(10):
            # for iter in range(1):
                ii = 0
                for inputs_SAR, inputs_EO, labels in test_data:
                    # tepoch.set_description(f"Epoch {epoch}")
                    # get the inputs
                    inputs_SAR, inputs_EO, labels = inputs_SAR.to(device), inputs_EO.to(device), labels.to(device)
                    # if inputs.shape[1] == 1:
                    #     inputs = inputs.repeat(1, 3, 1, 1)

                    # forward + get predictions + backward + optimize
                    # outputs_SAR = F(inputs_SAR)
                    outputs_EO = F(inputs_EO)
                    # outputs_SAR = outputs_SAR.view(outputs_SAR.shape[0], -1)
                    outputs_EO = outputs_EO.view(outputs_EO.shape[0], -1)
                    outputs = C(outputs_EO)

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    # outputs = torch.nn.functional.softmax(outputs, dim=-1)
                    # print(outputs)
                    # print(ii)
                    if ii == 0:
                        pred_outputs = outputs.cpu().detach()
                        labels_all = labels.cpu().detach()
                    else:
                        pred_outputs = torch.cat((pred_outputs, outputs.cpu().detach()), 0)
                        labels_all = torch.cat((labels_all, labels.cpu().detach()), 0)

                    ii += 1
                # print('Accuracy of the network on the ' + str(total) + ' test images: %d %%' % (100 * correct / total))
                acc = 100 * correct / total
                print('The acc of model {} is {}'.format(str(iter), acc))
                # print(pred_outputs.shape)
                all_pred_outputs[iter, :, :] = pred_outputs

        all_pred_outputs = torch.squeeze(torch.mean(all_pred_outputs, 0))
        _, predicted = torch.max(all_pred_outputs, 1)
        total = labels_all.size(0)
        correct = (predicted == labels_all).sum().item()
        acc = 100 * correct / total
        print('Current accuracy of the network on the ' + str(total) + ' test images: %d %%' % (acc))
        print('Best accuracy of the network on the ' + str(total) + ' test images: %d %%' % (best_acc))

        if acc > best_acc:
            best_acc = acc
            torch.save(F, 'swinB_F_bestacc_Track2_EO.pth')
            torch.save(C, 'swinB_C_bestacc_Track2_EO.pth')

    print('Finished Training')
    torch.save(F, 'swinB_F_final_Track2_EO.pth')
    torch.save(C, 'swinB_C_final_Track2_EO.pth')


def test(model_path):
    args = parse_args()
    test_data = load_test_data(args.test_folder, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    model = torch.load(model_path)
    model.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_data:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the ' + str(total) + ' test images: %d %%' % (
            100 * correct / total))


if __name__ == "__main__":
    args = parse_args()
    if args.test_only:
        test(args.model_path)
    else:
        train()
