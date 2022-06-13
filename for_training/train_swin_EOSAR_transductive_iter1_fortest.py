# coding: utf-8
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
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

from itertools import cycle

import csv
import scipy.io as sio
import sys

# 载入sklearn中的k近邻分类器方法
from sklearn.neighbors import KNeighborsClassifier


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
    # parser.add_argument('--batch_size', help=h['batch_size'], type=int, default=32)
    parser.add_argument('--batch_size', help=h['batch_size'], type=int, default=16)
    # parser.add_argument('--batch_size', help=h['batch_size'], type=int, default=8)
    # parser.add_argument('--batch_size', help=h['batch_size'], type=int, default=4)
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
        for dir_name in sorted(os.listdir(self.root_dir)):
            dir_files = []
            for file in sorted(os.listdir(self.root_dir + '/' + dir_name)):
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


class MyData4_forfeature(Dataset):  # 继承Dataset
    def __init__(self, root_dir, flag, transform_SAR=None, transform_EO=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.transform_SAR = transform_SAR  # 变换
        self.transform_EO = transform_EO  # 变换

        self.all_files_train = []
        self.all_files_test = []
        for dir_name in sorted(os.listdir(self.root_dir)):
            dir_files = []
            for file in sorted(os.listdir(self.root_dir + '/' + dir_name)):
                # print(os.path.join(root, file))
                if 'EO' not in file:
                    # self.all_files.append(self.root_dir + '/' + dir_name + '/' + file)
                    dir_files.append(self.root_dir + '/' + dir_name + '/' + file)   # append the SAR image
            random.seed(0)
            np.random.seed(0)
            random.shuffle(dir_files)
            test_files = dir_files[:100]
            # train_files = dir_files[100:]
            train_files = dir_files[100:600]

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

        self.targets = self.cal_concat_targets()
        # print(self.targets)

        # if self.flag == 1:
        #     self.c0 = np.where(np.array(self.targets)==0)[0].tolist()
        #     self.c1 = np.where(np.array(self.targets)==1)[0].tolist()
        #     self.c2 = np.where(np.array(self.targets)==2)[0].tolist()
        #     self.c3 = np.where(np.array(self.targets)==3)[0].tolist()
        #     self.c4 = np.where(np.array(self.targets)==4)[0].tolist()
        #     self.c5 = np.where(np.array(self.targets)==5)[0].tolist()
        #     self.c6 = np.where(np.array(self.targets)==6)[0].tolist()
        #     self.c7 = np.where(np.array(self.targets)==7)[0].tolist()
        #     self.c8 = np.where(np.array(self.targets)==8)[0].tolist()
        #     self.c9 = np.where(np.array(self.targets)==9)[0].tolist()
        #     # print(len(self.c0))
        # elif self.flag == 0:
        #     pass

    def __len__(self):  # 返回整个数据集的大小
        # return len(self.all_files)
        if self.flag == 1:
            # return 500 * 10
            return len(self.all_files)
        elif self.flag == 0:
            return len(self.all_files)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        # img_path = os.path.join(self.root_dir, image_index)  # 获取索引为index的图片的路径名
        # print(index)

        # if self.flag == 1:
        #
        #     random.seed(None)
        #     np.random.seed(None)
        #     c0 = random.sample(self.c0, 500)
        #     c1 = random.sample(self.c1, 500)
        #     c2 = random.sample(self.c2, 500)
        #     c3 = random.sample(self.c3, 500)
        #     c4 = random.sample(self.c4, 500)
        #     c5 = random.sample(self.c5, 500)
        #     c6 = random.sample(self.c6, 500)
        #     c7 = random.sample(self.c7, 500)
        #     c8 = random.sample(self.c8, 500)
        #     c9 = random.sample(self.c9, 500)
        #
        #     updating_train_list = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9
        #     updating_train_list = np.array(self.all_files)[updating_train_list].tolist()
        #
        #     # img_path = self.all_files[index]
        #     img_path = updating_train_list[index]
        #
        # elif self.flag == 0:
        #     img_path = self.all_files[index]

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


class MyData_test(Dataset):  # 继承Dataset
    def __init__(self, root_dir_SAR, root_dir_EO, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir_SAR = root_dir_SAR  # 文件目录
        self.root_dir_EO = root_dir_EO  # 文件目录
        self.transform = transform  # 变换

        self.all_files = []
        for file in sorted(os.listdir(self.root_dir_SAR)):
            # print(os.path.join(root, file))
            if 'EO' not in file:
                self.all_files.append(self.root_dir_SAR + '/' + file)

    def __len__(self):  # 返回整个数据集的大小
        return len(self.all_files)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        # img_path = os.path.join(self.root_dir, image_index)  # 获取索引为index的图片的路径名
        img_path_SAR = self.all_files[index]
        img_path_EO = img_path_SAR.replace('SAR', 'EO')
        # img = io.imread(img_path)  # 读取该图片
        img_SAR = Image.open(img_path_SAR)  # 读取该图片
        img_EO = Image.open(img_path_EO)  # 读取该图片
        img_SAR = img_SAR.convert('RGB')
        img_EO = img_EO.convert('RGB')
        # label = int(img_path.split('/')[-2])
        name = int(img_path_SAR.split('/')[-1].split('_')[-1].split('.')[0])   # read the number of the images
        # print(label)
        # sample = {'image': img, 'label': label}  # 根据图片和标签创建字典

        if self.transform:
            # sample = self.transform(sample)  # 对样本进行变换
            img_SAR = self.transform(img_SAR)  # 对样本进行变换
            img_EO = self.transform(img_EO)  # 对样本进行变换
        # return sample  # 返回该样本
        # return img, label  # 返回该样本
        # return img, name  # 返回该样本
        return img_SAR, img_EO, name  # 返回该样本


class MyData_test_new(Dataset):  # 继承Dataset
    def __init__(self, root_dir_SAR, root_dir_EO, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir_SAR = root_dir_SAR  # 文件目录
        self.root_dir_EO = root_dir_EO  # 文件目录
        self.transform = transform  # 变换

        self.all_files = []
        for file in sorted(os.listdir(self.root_dir_SAR)):
            # print(os.path.join(root, file))
            if 'EO' not in file:
                self.all_files.append(self.root_dir_SAR + '/' + file)

    def __len__(self):  # 返回整个数据集的大小
        return len(self.all_files)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        # img_path = os.path.join(self.root_dir, image_index)  # 获取索引为index的图片的路径名
        img_path_SAR = self.all_files[index]
        img_path_EO = img_path_SAR.replace('SAR', 'EO')
        # img = io.imread(img_path)  # 读取该图片
        img_SAR = Image.open(img_path_SAR)  # 读取该图片
        img_EO = Image.open(img_path_EO)  # 读取该图片
        img_SAR = img_SAR.convert('RGB')
        img_EO = img_EO.convert('RGB')
        # label = int(img_path.split('/')[-2])
        name = int(img_path_SAR.split('/')[-1].split('_')[-1].split('.')[0])   # read the number of the images
        # print(label)
        # sample = {'image': img, 'label': label}  # 根据图片和标签创建字典

        if self.transform:
            # sample = self.transform(sample)  # 对样本进行变换
            img_SAR = self.transform(img_SAR)  # 对样本进行变换
            img_EO = self.transform(img_EO)  # 对样本进行变换
        # return sample  # 返回该样本
        # return img, label  # 返回该样本
        # return img, name  # 返回该样本
        # return img_SAR, img_EO, name  # 返回该样本
        return img_SAR, img_EO, index, name  # 返回该样本


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

    train_data_loader = torch.utils.data.DataLoader(train_data_raw_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    # test_data_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    test_data_loader = torch.utils.data.DataLoader(train_data_raw_test, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

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
    # transform_train_SAR = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(10, resample=False, expand=False, center=None), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transform_train_EO = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(10, resample=False, expand=False, center=None), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transform_train_SAR = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(10, resample=False, expand=False, center=None), transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transform_train_EO = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(10, resample=False, expand=False, center=None), transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transform_train_SAR = transforms.Compose([transforms.Resize(480), transforms.RandomCrop(448), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(180, resample=False, expand=False, center=None), transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transform_train_EO = transforms.Compose([transforms.Resize(480), transforms.RandomCrop(448), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(180, resample=False, expand=False, center=None), transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_train_SAR = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(180, resample=False, expand=False, center=None), transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_train_EO = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(180, resample=False, expand=False, center=None), transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
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

    train_data_loader = torch.utils.data.DataLoader(train_data_raw_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    # test_data_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    # mean_SAR, std_SAR, mean_EO, std_EO = get_mean_std_value(train_data_loader)
    # print(mean_SAR, std_SAR, mean_EO, std_EO)

    test_data_loader = torch.utils.data.DataLoader(train_data_raw_test, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    return train_data_loader, test_data_loader


def load_train_myval_pairdata_forfeature(train_data_path, batch_size):
    # Convert images to tensors, normalize, and resize them
    # transform_train_SAR = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(180, resample=False, expand=False, center=None), transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transform_train_EO = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(180, resample=False, expand=False, center=None), transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_train_SAR = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(10, resample=False, expand=False, center=None), transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_train_EO = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(10, resample=False, expand=False, center=None), transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transform_test_SAR = transform_train_SAR
    # transform_test_EO = transform_train_EO
    train_data_raw_train = MyData4_forfeature(train_data_path, flag=1, transform_SAR=transform_train_SAR, transform_EO=transform_train_EO)  # 初始化类，设置数据集所在路径以及变换
    # train_data_raw_test = MyData4_forfeature(train_data_path, flag=0, transform_SAR=transform_test_SAR, transform_EO=transform_test_EO)  # 初始化类，设置数据集所在路径以及变换

    # train_data_loader = torch.utils.data.DataLoader(train_data_raw_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    train_data_loader = torch.utils.data.DataLoader(train_data_raw_train, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    # test_data_loader = torch.utils.data.DataLoader(train_data_raw_test, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    return train_data_loader


# def load_test_data(test_data_path, batch_size):
#     # transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#     transform = transforms.Compose(transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(10, resample=False, expand=False, center=None), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
#
#     test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=transform)
#     # test_data_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
#     test_data_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
#
#     # return test_data_loader
#     return test_data, test_data_loader


def load_test_data(test_data_path_SAR, test_data_path_EO, batch_size):
    # transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transform = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.ToTensor(), transforms.Normalize((0.3984, 0.3984, 0.3984), (0.1328, 0.1328, 0.1328))])
    # transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.2930, 0.2930, 0.2930), (0.1479, 0.1479, 0.1479))])
    # transform = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.ToTensor(), transforms.Normalize((0.2930, 0.2930, 0.2930), (0.1479, 0.1479, 0.1479))])
    # transform = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.ToTensor(), transforms.Normalize((0.2930, 0.2930, 0.2930), (0.1479, 0.1479, 0.1479))])
    # transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_test = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(10, resample=False, expand=False, center=None), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transform_train = transforms.Compose([transforms.Resize(480), transforms.RandomCrop(448), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(180, resample=False, expand=False, center=None), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_train = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(180, resample=False, expand=False, center=None), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=transform)
    # test_data = MyData_test(test_data_path_SAR, test_data_path_EO, transform=transform)
    test_data = MyData_test_new(test_data_path_SAR, test_data_path_EO, transform=transform_test)
    # test_data_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    test_data_train = MyData_test_new(test_data_path_SAR, test_data_path_EO, transform=transform_train)

    return test_data, test_data_loader, test_data_train


def load_test_data_iterative(test_data_path_SAR, test_data_path_EO, batch_size):
    # transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transform = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.ToTensor(), transforms.Normalize((0.3984, 0.3984, 0.3984), (0.1328, 0.1328, 0.1328))])
    # transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.2930, 0.2930, 0.2930), (0.1479, 0.1479, 0.1479))])
    # transform = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.ToTensor(), transforms.Normalize((0.2930, 0.2930, 0.2930), (0.1479, 0.1479, 0.1479))])
    # transform = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.ToTensor(), transforms.Normalize((0.2930, 0.2930, 0.2930), (0.1479, 0.1479, 0.1479))])
    # transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_test = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(180, resample=False, expand=False, center=None), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transform_train = transforms.Compose([transforms.Resize(480), transforms.RandomCrop(448), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(180, resample=False, expand=False, center=None), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transform_train = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.RandomRotation(180, resample=False, expand=False, center=None), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=transform)
    # test_data = MyData_test(test_data_path_SAR, test_data_path_EO, transform=transform)
    test_data = MyData_test_new(test_data_path_SAR, test_data_path_EO, transform=transform_test)
    # test_data_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    # test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    # test_data_train = MyData_test_new(test_data_path_SAR, test_data_path_EO, transform=transform_train)

    return test_data


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


def set_weight_decay(model):
    has_decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def my_gather_features(netF, dataloader):
    F = []
    # Y = []
    for i, data in enumerate(dataloader):
        images_SAR, images_EO, _, _ = data
        if images_EO.shape[1] == 3:
            pass
        else:
            images_EO = images_EO.repeat(1, 3, 1, 1)
        images_EO = images_EO.cuda()
        features = netF(images_EO)
        features = features.view(features.shape[0], -1)
        F += features.cpu().detach().tolist()
        # Y += labels.tolist()
    F = np.asarray(F)
    # Y = np.asarray(Y)
    return F


def my_gather_features_with_label(netF, dataloader):
    F = []
    Y = []
    for i, data in enumerate(dataloader):
        images_SAR, images_EO, labels = data
        if images_EO.shape[1] == 3:
            pass
        else:
            images_EO = images_EO.repeat(1, 3, 1, 1)
        images_EO = images_EO.cuda()
        features = netF(images_EO)
        features = features.view(features.shape[0], -1)
        F += features.cpu().detach().tolist()
        Y += labels.tolist()
    F = np.asarray(F)
    Y = np.asarray(Y)
    return F, Y


def my_gather_features_SAR(netF, dataloader):
    F = []
    # Y = []
    for i, data in enumerate(dataloader):
        images_SAR, images_EO, _, _ = data
        if images_SAR.shape[1] == 3:
            pass
        else:
            images_SAR = images_SAR.repeat(1, 3, 1, 1)
        images_SAR = images_SAR.cuda()
        features = netF(images_SAR)
        features = features.view(features.shape[0], -1)
        F += features.cpu().detach().tolist()
        # Y += labels.tolist()
    F = np.asarray(F)
    # Y = np.asarray(Y)
    return F


def my_gather_features_SAR_with_label(netF, dataloader):
    F = []
    Y = []
    for i, data in enumerate(dataloader):
        images_SAR, images_EO, labels = data
        if images_SAR.shape[1] == 3:
            pass
        else:
            images_SAR = images_SAR.repeat(1, 3, 1, 1)
        images_SAR = images_SAR.cuda()
        features = netF(images_SAR)
        features = features.view(features.shape[0], -1)
        F += features.cpu().detach().tolist()
        Y += labels.tolist()
    F = np.asarray(F)
    Y = np.asarray(Y)
    return F, Y


def train():
    args = parse_args()
    # train_data = load_train_data(args.train_folder, args.batch_size)
    # train_data, test_data = load_train_myval_data(args.train_folder, args.batch_size)
    # train_data, test_data = load_train_myval_pairdata(args.train_folder, args.batch_size)
    test_data_path_SAR = '../test_images_SAR'
    test_data_path_EO = '../test_images_EO'
    # test_data = load_test_data(test_data_path_SAR, test_data_path_EO, args.batch_size)
    test_data, test_data_loader, test_data_train = load_test_data(test_data_path_SAR, test_data_path_EO, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ########################### predict by the swinB_F/C_bestacc_Track2_EO.pth ##################################
    # F = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=10,
    F = SwinTransformer(img_size=224, patch_size=4, in_chans=2, num_classes=10,
                        embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                        window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                        # window_size=14, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                        # use_checkpoint=False
                        use_checkpoint=True
                        )  # the feature dim is 1024
    F.apply(inplace_relu)
    # F = torch.load('swinB_F_bestacc_Track2_EO4_transductive.pth')
    # F = torch.load('swinB_F_bestacc_Track2_EO.pth')
    F = torch.load('swinB_F_bestacc_Track2_EOSAR_fortest.pth')
    # F = torch.load('swinB_F_bestacc_Track2_EO5_transductive_iter1.pth')
    F = F.cuda()
    C = Classifier(10)
    # C = torch.load('swinB_C_bestacc_Track2_EO4_transductive.pth')
    # C = torch.load('swinB_C_bestacc_Track2_EO.pth')
    C = torch.load('swinB_C_bestacc_Track2_EOSAR_fortest.pth')
    # C = torch.load('swinB_C_bestacc_Track2_EO5_transductive_iter1.pth')
    C.apply(inplace_relu)
    C = C.cuda()
    F.eval()
    C.eval()
    for i in F.parameters():
        i.requires_grad = False
    for i in C.parameters():
        i.requires_grad = False

    with torch.no_grad():
        all_pred_outputs = torch.zeros(10, 826, 10)
        # all_pred_outputs = torch.zeros(1, 826, 10)
        for iter in range(10):
        # for iter in range(1):
            ii = 0
            # for inputs, labels in test_data:
            # for inputs_SAR, inputs_EO, name in test_data:
            # for inputs_SAR, inputs_EO, name in test_data_loader:
            for inputs_SAR, inputs_EO, _, _ in test_data_loader:
                # tepoch.set_description(f"Epoch {epoch}")
                # get the inputs
                inputs_SAR = inputs_SAR.to(device)
                inputs_EO = inputs_EO.to(device)
                inputs = torch.cat((inputs_EO[:, 0, :, :].unsqueeze(1), inputs_SAR[:, 0, :, :].unsqueeze(1)), 1)
                # if inputs.shape[1] == 1:
                #     inputs = inputs.repeat(1, 3, 1, 1)

                # forward + get predictions + backward + optimize
                # outputs_SAR = F(inputs_SAR)
                # outputs_EO = F(inputs_EO)
                outputs = F(inputs)
                # outputs_SAR = outputs_SAR.view(outputs_SAR.shape[0], -1)
                # outputs_EO = outputs_EO.view(outputs_EO.shape[0], -1)
                outputs = outputs.view(outputs.shape[0], -1)
                # outputs = C(outputs_EO)
                outputs = C(outputs)

                _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()

                outputs = torch.nn.functional.softmax(outputs, dim=-1)
                # print(outputs)
                # print(ii)
                if ii == 0:
                    pred_outputs = outputs.cpu().detach()
                    # labels_all = labels.cpu().detach()
                    # name_all = name.cpu().detach()
                else:
                    pred_outputs = torch.cat((pred_outputs, outputs.cpu().detach()), 0)
                    # labels_all = torch.cat((labels_all, labels.cpu().detach()), 0)
                    # name_all = torch.cat((name_all, name.cpu().detach()), 0)

                ii += 1
                print((ii - 1) * args.batch_size + inputs_SAR.shape[0])

            # print('Accuracy of the network on the ' + str(total) + ' test images: %d %%' % (100 * correct / total))
            # acc = 100 * correct / total
            # print('The acc of model {} is {}'.format(str(iter), acc))
            # print(pred_outputs.shape)
            all_pred_outputs[iter, :, :] = pred_outputs   # [10, 770, 10]

    # sio.savemat('EO10_fortest_bestacc.mat', {'preds': all_pred_outputs.numpy()})
    # sys.exit(0)

    all_pred_outputs = torch.squeeze(torch.mean(all_pred_outputs, 0))  # [770, 10]
    max_outputs, argmax_preds = torch.max(all_pred_outputs, 1)
    max_outputs = torch.squeeze(max_outputs)   # [770]
    preds1 = torch.squeeze(argmax_preds)   # [770]

    selected_list_ori = np.where(max_outputs.numpy() > 0.9)[0].tolist()
    selected_max_outputs = max_outputs[selected_list_ori]
    sorted_maxprob, indices_maxprob = torch.sort(selected_max_outputs, descending=True)   # from the larger to smaller
    selected_indices = indices_maxprob[:max(200, int(0.5 * len(indices_maxprob)))]
    # selected_indices = indices_maxprob[:400]   # selected more samples
    selected_list1 = np.array(selected_list_ori)[selected_indices].tolist()

    # ############ in the feature space of the pretrained weights ################
    F0 = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                         embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                         window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                         drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                         norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                         # use_checkpoint=False
                         use_checkpoint=True
                         )  # the feature dim is 1024
    F0.load_state_dict(torch.load('swin_base_patch4_window7_224_22k.pth')['model'], strict=False)  # use the pretrained model trained on ImageNet
    F0.apply(inplace_relu)
    F0 = F0.cuda()
    F0.eval()
    for p in F0.parameters():
        p.requires_grad_(False)  # freeze F0

    with torch.no_grad():
        # F0test = my_gather_features_SAR(F0, test_data_loader)
        F0test = my_gather_features(F0, test_data_loader)
    F0test_selected = F0test[selected_list1]

    train_data_loader_part = load_train_myval_pairdata_forfeature(args.train_folder, args.batch_size)
    with torch.no_grad():
        # F0train_part, ltrain_part = my_gather_features_SAR_with_label(F0, train_data_loader_part)
        F0train_part, ltrain_part = my_gather_features_with_label(F0, train_data_loader_part)

    knn = KNeighborsClassifier(n_neighbors=10, p=2)  # p=2表示欧氏距离, 10 nearest neighbors
    knn.fit(F0train_part, ltrain_part)
    # pred_selected = knn.predict(F0test_selected)
    preds_all = knn.predict(F0test)
    # dist_10, indices_10 = knn.kneighbors(F0test_selected, n_neighbors=10, return_distance=True)
    # pred_10 = ltrain_part[indices_10]

    # print(pred_selected)
    preds1 = preds_all
    print(preds1.shape)
    preds1 = torch.from_numpy(preds1)


    # # 开始判断，判断准则：先数量，后距离，同时计算置信度
    # test_list_final = copy.deepcopy(selected_list1)
    # for i in range(F0test_selected.shape[0]):
    #     # print(i)
    #     pred_this_sample = pred_10[i, :]
    #     the_count = np.bincount(pred_this_sample)
    #     # sorted_value, sorted_index = torch.sort(the_count)   # 从小到大排序，返回排序后的值和索引
    #     sorted_value = np.sort(the_count)  # 从小到大排序，返回排序后的值和索引
    #     sorted_index = np.argsort(the_count)  # 从小到大排序，返回排序后的值和索引
    #     sorted_value = sorted_value.tolist()
    #     sorted_index = sorted_index.tolist()
    #     if len(sorted_value) > 1:
    #         if sorted_value[-1] != sorted_value[-2]:  # 邻域内数量最多的类别
    #             pred_label = sorted_index[-1]
    #         else:  # 假设邻域内有多个类别数量相同且最多，则分为最近的类别
    #             # print(sorted_index)
    #             # print(sorted_value)
    #             most_indices = sorted_index[sorted_index.index(sorted_value == sorted_value[-1]):]
    #             # print(dist_10_known)
    #             # print(most_indices)
    #             their_dists = dist_10_known[i, :][most_indices]
    #             pred_label = most_indices[np.argmin(their_dists)]
    #     else:
    #         pred_label = sorted_index[-1]
    #     if pred_label == 0:  # 若仍判别为已知类别
    #         pass
    #     elif pred_label == 1:  # 若筛选出的已知类别样本在这里被判别为未知类别
    #         known_test_list_final.remove(known_test_list[i])  # 从筛选出的已知类别列表中删除这个样本
    #     # unknown_test_list_final.append(known_test_list[i])  # 从筛选出的未知类别列表中添加这个样本
    #     else:
    #         known_test_list_final.remove(known_test_list[i])  # 从筛选出的已知类别列表中删除这个样本



    # ##################### predict by the swinB_F/C_bestacc_Track2_SAR.pth  ############################################
    # test_data_train_loader = torch.utils.data.DataLoader(test_data_train, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # F1 = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=10,
    # # F1 = SwinTransformer(img_size=448, patch_size=4, in_chans=3, num_classes=10,
    #                     embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
    #                     window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
    #                     # window_size=14, mlp_ratio=4., qkv_bias=True, qk_scale=None,
    #                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
    #                     norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
    #                     # use_checkpoint=False
    #                     use_checkpoint=True
    #                     )  # the feature dim is 1024
    # F1.apply(inplace_relu)
    # F1 = torch.load('swinB_F_bestacc_Track2_SAR10_fortest.pth')
    #
    # F1 = F1.cuda()
    # C1 = Classifier(10)
    # C1 = torch.load('swinB_C_bestacc_Track2_SAR10_fortest.pth')
    # C1.apply(inplace_relu)
    # C1 = C1.cuda()
    #
    # F1.eval()
    # C1.eval()
    # # correct = 0
    # # total = 0
    # with torch.no_grad():
    #     all_pred_outputs = torch.zeros(10, 826, 10)
    #     # all_pred_outputs = torch.zeros(1, 826, 10)
    #     for iter in range(10):
    #     # for iter in range(1):
    #         ii = 0
    #         for inputs_SAR, inputs_EO, _, name in test_data_train_loader:
    #             # tepoch.set_description(f"Epoch {epoch}")
    #             # get the inputs
    #             inputs_SAR, inputs_EO, name = inputs_SAR.to(device), inputs_EO.to(device), name.to(device)
    #             # if inputs.shape[1] == 1:
    #             #     inputs = inputs.repeat(1, 3, 1, 1)
    #
    #             # forward + get predictions + backward + optimize
    #             outputs_SAR = F1(inputs_SAR)
    #             # outputs_EO = F1(inputs_EO)
    #             outputs_SAR = outputs_SAR.view(outputs_SAR.shape[0], -1)
    #             # outputs_EO = outputs_EO.view(outputs_EO.shape[0], -1)
    #             outputs = C1(outputs_SAR)
    #             # outputs = C1(outputs_EO)
    #
    #             _, predicted = torch.max(outputs.data, 1)
    #             # total += labels.size(0)
    #             # correct += (predicted == labels).sum().item()
    #
    #             outputs = torch.nn.functional.softmax(outputs, dim=-1)
    #             # print(outputs)
    #             # print(ii)
    #             if ii == 0:
    #                 pred_outputs = outputs.cpu().detach()
    #                 name_all = name.cpu().detach()
    #             else:
    #                 pred_outputs = torch.cat((pred_outputs, outputs.cpu().detach()), 0)
    #                 name_all = torch.cat((name_all, name.cpu().detach()), 0)
    #
    #             ii += 1
    #         # print('Accuracy of the network on the ' + str(total) + ' test images: %d %%' % (100 * correct / total))
    #         # acc = 100 * correct / total
    #         # print('The acc of model {} is {}'.format(str(iter), acc))
    #         # print(pred_outputs.shape)
    #         all_pred_outputs[iter, :, :] = pred_outputs
    #
    # all_pred_outputs = torch.squeeze(torch.mean(all_pred_outputs, 0))  # [770, 10]
    # max_outputs, argmax_preds = torch.max(all_pred_outputs, 1)
    # max_outputs = torch.squeeze(max_outputs)   # [770]
    # preds2 = torch.squeeze(argmax_preds)   # [770]
    #
    # selected_list_ori = np.where(max_outputs.numpy() > 0.9)[0].tolist()
    # selected_max_outputs = max_outputs[selected_list_ori]
    # sorted_maxprob, indices_maxprob = torch.sort(selected_max_outputs, descending=True)   # from the larger to smaller
    # selected_indices = indices_maxprob[:max(200, int(0.5 * len(indices_maxprob)))]
    # selected_list2 = np.array(selected_list_ori)[selected_indices].tolist()
    #
    # # ############ in the feature space of the pretrained weights ################
    # F0 = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=1000,
    #                      embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
    #                      window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
    #                      drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
    #                      norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
    #                      # use_checkpoint=False
    #                      use_checkpoint=True
    #                      )  # the feature dim is 1024
    # F0.load_state_dict(torch.load('swin_base_patch4_window7_224_22k.pth')['model'], strict=False)  # use the pretrained model trained on ImageNet
    # F0.apply(inplace_relu)
    # F0 = F0.cuda()
    # F0.eval()
    # for p in F0.parameters():
    #     p.requires_grad_(False)  # freeze F0
    #
    # with torch.no_grad():
    #     F0test = my_gather_features_SAR(F0, test_data_loader)
    # F0test_selected = F0test[selected_list2]
    #
    # train_data_loader_part = load_train_myval_pairdata_forfeature(args.train_folder, args.batch_size)
    # with torch.no_grad():
    #     F0train_part, ltrain_part = my_gather_features_SAR_with_label(F0, train_data_loader_part)
    #
    # knn = KNeighborsClassifier(n_neighbors=10, p=2)  # p=2表示欧氏距离, 10 nearest neighbors
    # knn.fit(F0train_part, ltrain_part)
    # # pred_selected = knn.predict(F0test_selected)
    # preds_all = knn.predict(F0test)
    # # dist_10, indices_10 = knn.kneighbors(F0test_selected, n_neighbors=10, return_distance=True)
    # # pred_10 = ltrain_part[indices_10]
    #
    # # print(pred_selected)
    # preds2 = preds_all
    #
    # preds2 = torch.from_numpy(preds2)

    # ##################################################################################################################

    # rest_list = list(set(range(826)) - set(selected_list))

    iteration_num = 0
    # while len(selected_list_ori) > 100:
    while iteration_num < 1:
        iteration_num += 1

        # availabletesttransset = torch.utils.data.Subset(test_data, selected_list)
        availabletesttransset1 = torch.utils.data.Subset(test_data_train, selected_list1)
        # availabletesttransset2 = torch.utils.data.Subset(test_data_train, selected_list2)
        # test_data_loader = torch.utils.data.DataLoader(availabletesttransset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        # test_data_loader = torch.utils.data.DataLoader(availabletesttransset, batch_size=int(args.batch_size / 1), shuffle=True, num_workers=2)
        # test_data_loader = torch.utils.data.DataLoader(availabletesttransset, batch_size=int(args.batch_size / 8), shuffle=True, num_workers=2)
        test_data_loader1 = torch.utils.data.DataLoader(availabletesttransset1, batch_size=int(args.batch_size / 4), shuffle=True, num_workers=2)
        # test_data_loader2 = torch.utils.data.DataLoader(availabletesttransset2, batch_size=int(args.batch_size / 4), shuffle=True, num_workers=2)

        torch.cuda.empty_cache()

        # F1 = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=10,
        F1 = SwinTransformer(img_size=224, patch_size=4, in_chans=2, num_classes=10,
        # F1 = SwinTransformer(img_size=448, patch_size=4, in_chans=3, num_classes=10,
                            embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                            window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                            # window_size=14, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                            norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                            # use_checkpoint=False
                            use_checkpoint=True
                            )  # the feature dim is 1024
        F1.apply(inplace_relu)
        if iteration_num == 1:
            # F1 = torch.load('swinB_F_bestacc_Track2_EO.pth')
            F1 = torch.load('swinB_F_bestacc_Track2_EOSAR_fortest.pth')
        else:
            F1 = torch.load('swinB_F_bestacc_Track2_EO10_iteration_num{}_fortest.pth'.format(iteration_num-1))
        # F1 = torch.load('swinB_F_bestacc_Track2_EO5_transductive_iter1.pth')

        F1 = F1.cuda()
        C1 = Classifier(10)
        if iteration_num ==1:
            # C1 = torch.load('swinB_C_bestacc_Track2_EO.pth')
            C1 = torch.load('swinB_C_bestacc_Track2_EOSAR_fortest.pth')
        else:
            C1 = torch.load('swinB_C_bestacc_Track2_EO10_iteration_num{}_fortest.pth'.format(iteration_num-1))
        # C1 = torch.load('swinB_C_bestacc_Track2_EO5_transductive_iter1.pth')
        C1.apply(inplace_relu)
        C1 = C1.cuda()
        F1.train()
        C1.train()

        train_data_loader, val_data_loader = load_train_myval_pairdata(args.train_folder, args.batch_size)

        criterion = nn.CrossEntropyLoss()

        F1_parameters = set_weight_decay(F1)

        # optimizer_F1 = optim.AdamW(F1.parameters(), lr=0.00002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        optimizer_F1 = optim.AdamW(F1_parameters, lr=0.00002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        optimizer_C1 = optim.SGD(C1.parameters(), lr=0.001, weight_decay=1e-4)

        best_acc = 0.

        # for epoch in range(args.epochs):  # loop over the dataset multiple times
        for epoch in range(8):  # loop over the dataset multiple times
        # for epoch in range(1):  # loop over the dataset multiple times
            # with tqdm(train_data, unit="batch") as tepoch:
            #     for inputs, labels in tepoch:
            F1.train()
            C1.train()
            # for inputs, labels in train_data:
            # train_data, _ = load_train_myval_data(args.train_folder, args.batch_size)
            # for batch_idx, data in enumerate(zip(train_data_loader, cycle(test_data_loader))):
            # for batch_idx, data in enumerate(zip(train_data_loader, cycle(test_data_loader1), cycle(test_data_loader2))):
            for batch_idx, data in enumerate(zip(train_data_loader, cycle(test_data_loader1))):

                inputs_SAR, inputs_EO, train_labels = data[0]
                # inputs_SAR_test, inputs_EO_test, _ = data[1]
                inputs_SAR_test1, inputs_EO_test1, indices1, _ = data[1]
                # inputs_SAR_test2, inputs_EO_test2, indices2, _ = data[2]

                inputs_SAR, inputs_EO, labels = inputs_SAR.to(device), inputs_EO.to(device), train_labels.to(device)
                # if inputs.shape[1] == 1:
                #     inputs = inputs.repeat(1, 3, 1, 1)

                # zero the parameter gradients
                # optimizer.zero_grad()
                optimizer_F1.zero_grad()
                optimizer_C1.zero_grad()

                # ############### predict the pseudo labels of the selected test samples #######################
                inputs_EO_test1 = inputs_EO_test1.to(device)
                inputs_SAR_test1 = inputs_SAR_test1.to(device)
                # inputs_EO_test2 = inputs_EO_test2.to(device)
                # with torch.no_grad():
                    # labels_EO_test = F(inputs_EO_test)
                    # labels_EO_test = labels_EO_test.view(labels_EO_test.shape[0], -1)
                    # labels_EO_test = C(labels_EO_test)
                    # _, labels_EO_test = torch.max(labels_EO_test.data, 1)
                # ####################### new try ###############################
                # print(indices)
                # print(indices.shape)
                # labels_EO_test = selected_labels[indices.tolist()]
                labels_EO_test1 = preds1[indices1.tolist()]
                labels_EO_test1 = labels_EO_test1.cuda()
                # labels_EO_test2 = preds2[indices2.tolist()]
                # labels_EO_test2 = labels_EO_test2.cuda()

                # ##############################################################################################

                # forward + get predictions + backward + optimize
                # outputs_SAR = F(inputs_SAR)
                inputs = torch.cat((inputs_EO[:, 0, :, :].unsqueeze(1), inputs_SAR[:, 0, :, :].unsqueeze(1)), 1)
                # outputs_EO = F1(inputs_EO)
                outputs = F1(inputs)
                # outputs_SAR = outputs_SAR.view(outputs_SAR.shape[0], -1)
                # outputs_EO = outputs_EO.view(outputs_EO.shape[0], -1)
                outputs = outputs.view(outputs.shape[0], -1)
                # outputs = C1(outputs_EO)
                outputs = C1(outputs)
                loss1 = criterion(outputs, labels)
                inputs_test1 = torch.cat((inputs_EO_test1[:, 0, :, :].unsqueeze(1), inputs_SAR_test1[:, 0, :, :].unsqueeze(1)), 1)
                # outputs_EO_test1 = F1(inputs_EO_test1)
                outputs_test1 = F1(inputs_test1)
                # outputs_EO_test1 = outputs_EO_test1.view(outputs_EO_test1.shape[0], -1)
                outputs_test1 = outputs_test1.view(outputs_test1.shape[0], -1)
                # outputs_test1 = C1(outputs_EO_test1)
                outputs_test1 = C1(outputs_test1)
                loss2_1 = criterion(outputs_test1, labels_EO_test1)
                # outputs_EO_test2 = F1(inputs_EO_test2)
                # outputs_EO_test2 = outputs_EO_test2.view(outputs_EO_test2.shape[0], -1)
                # outputs_test2 = C1(outputs_EO_test2)
                # loss2_2 = criterion(outputs_test2, labels_EO_test2)

                # loss = loss1 + loss2
                # loss = loss1 + 0.5 * loss2_1 + 0.5 * loss2_2
                loss = loss1 + 0.5 * loss2_1
                # loss = loss1 + 0.1 * loss2
                # loss = loss1 + 0.5 * loss2

                predictions1 = outputs.argmax(dim=1, keepdim=True).squeeze()
                predictions2_1 = outputs_test1.argmax(dim=1, keepdim=True).squeeze()
                # predictions2_2 = outputs_test2.argmax(dim=1, keepdim=True).squeeze()
                # print(predictions)
                # print(labels)
                correct1 = (predictions1 == labels).sum().item()
                accuracy1 = correct1 / inputs_EO.shape[0]
                correct2_1 = (predictions2_1 == labels_EO_test1).sum().item()
                accuracy2_1 = correct2_1 / inputs_EO_test1.shape[0]
                # correct2_2 = (predictions2_2 == labels_EO_test2).sum().item()
                # accuracy2_2 = correct2_2 / inputs_EO_test2.shape[0]

                # print(train_labels)
                # print(predictions1)
                # print(labels_EO_test)
                # print(predictions2)

                loss.backward()
                optimizer_F1.step()
                optimizer_C1.step()

                # tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
                # print('loss {:.4f}  acc {:.4f}'.format(loss.item(), 100. * accuracy))
                # print('loss {:.4f} = loss1 {:.4f} + loss2 {:.4f} ;  acc1 {:.4f} acc2 {:.4f}'.format(loss.item(), loss1.item(), loss2.item(), 100. * accuracy1, 100. * accuracy2))
                # print('loss {:.4f} = loss1 {:.4f} + loss2_1 {:.4f}  + loss2_2 {:.4f} ;  acc1 {:.4f} acc2_1 {:.4f} acc2_2 {:.4f}'.format(loss.item(), loss1.item(), loss2_1.item(), loss2_2.item(), 100. * accuracy1, 100. * accuracy2_1, 100. * accuracy2_2))
                print('loss {:.4f} = loss1 {:.4f} + loss2_1 {:.4f} ;  acc1 {:.4f} acc2_1 {:.4f} '.format(loss.item(), loss1.item(), loss2_1.item(), 100. * accuracy1, 100. * accuracy2_1))

            F1.eval()
            C1.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                all_pred_outputs = torch.zeros(10, 1000, 10)
                # all_pred_outputs = torch.zeros(1, 1000, 10)
                for iter in range(10):
                # for iter in range(1):
                    ii = 0
                    for inputs_SAR, inputs_EO, labels in val_data_loader:
                        # tepoch.set_description(f"Epoch {epoch}")
                        # get the inputs
                        inputs_SAR, inputs_EO, labels = inputs_SAR.to(device), inputs_EO.to(device), labels.to(device)
                        # if inputs.shape[1] == 1:
                        #     inputs = inputs.repeat(1, 3, 1, 1)

                        # forward + get predictions + backward + optimize
                        inputs = torch.cat((inputs_EO[:, 0, :, :].unsqueeze(1), inputs_SAR[:, 0, :, :].unsqueeze(1)), 1)
                        # outputs_SAR = F(inputs_SAR)
                        # outputs_EO = F1(inputs_EO)
                        outputs = F1(inputs)
                        # outputs_SAR = outputs_SAR.view(outputs_SAR.shape[0], -1)
                        # outputs_EO = outputs_EO.view(outputs_EO.shape[0], -1)
                        outputs = outputs.view(outputs.shape[0], -1)
                        # outputs = C1(outputs_EO)
                        outputs = C1(outputs)

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
                torch.save(F1, 'swinB_F_bestacc_Track2_EOSAR_iteration_num{}_fortest.pth'.format(iteration_num))
                torch.save(C1, 'swinB_C_bestacc_Track2_EOSAR_iteration_num{}_fortest.pth'.format(iteration_num))

            torch.save(F1, 'swinB_F_now_Track2_EOSAR_iteration_num{}_fortest.pth'.format(iteration_num))
            torch.save(C1, 'swinB_C_now_Track2_EOSAR_iteration_num{}_fortest.pth'.format(iteration_num))

        print('Finished Training')
        torch.save(F1, 'swinB_F_final_Track2_EOSAR_iteration_num{}_fortest.pth'.format(iteration_num))
        torch.save(C1, 'swinB_C_final_Track2_EOSAR_iteration_num{}_fortest.pth'.format(iteration_num))

        sys.exit(0)

        # #################################################################################################
        # ####################### Please note that the following codes do not work ########################
        ###################################################################################################

        # ################################### begin new prediction ########################################
        test_data_ = load_test_data_iterative(test_data_path_SAR, test_data_path_EO, args.batch_size)
        test_data = torch.utils.data.Subset(test_data_, rest_list)
        test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False, num_workers=args.num_workers)

        F = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=10,
                            embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                            window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                            # window_size=14, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                            norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                            # use_checkpoint=False
                            use_checkpoint=True
                            )  # the feature dim is 1024
        F.apply(inplace_relu)
        # F = torch.load('swinB_F_bestacc_Track2_EO4_transductive.pth')
        F = torch.load('swinB_F_bestacc_Track2_EO10_iteration_num{}_fortest.pth'.format(iteration_num))
        # F = torch.load('swinB_F_bestacc_Track2_EO5_transductive_iter1.pth')
        F = F.cuda()
        C = Classifier(10)
        # C = torch.load('swinB_C_bestacc_Track2_EO4_transductive.pth')
        C = torch.load('swinB_C_bestacc_Track2_EO10_iteration_num{}_fortest.pth'.format(iteration_num))
        # C = torch.load('swinB_C_bestacc_Track2_EO5_transductive_iter1.pth')
        C.apply(inplace_relu)
        C = C.cuda()
        F.eval()
        C.eval()
        for i in F.parameters():
            i.requires_grad = False
        for i in C.parameters():
            i.requires_grad = False

        with torch.no_grad():
            all_pred_outputs = torch.zeros(10, len(rest_list), 10)
            # all_pred_outputs = torch.zeros(1, len(rest_list), 10)
            for iter in range(10):
            # for iter in range(1):
                ii = 0
                # for inputs, labels in test_data:
                # for inputs_SAR, inputs_EO, name in test_data:
                # for inputs_SAR, inputs_EO, name in test_data_loader:
                for inputs_SAR, inputs_EO, _, _ in test_data_loader:
                    # tepoch.set_description(f"Epoch {epoch}")
                    # get the inputs
                    # inputs_SAR = inputs_SAR.to(device)
                    inputs_EO = inputs_EO.to(device)
                    # if inputs.shape[1] == 1:
                    #     inputs = inputs.repeat(1, 3, 1, 1)

                    # forward + get predictions + backward + optimize
                    # outputs_SAR = F(inputs_SAR)
                    outputs_EO = F(inputs_EO)
                    # outputs_SAR = outputs_SAR.view(outputs_SAR.shape[0], -1)
                    outputs_EO = outputs_EO.view(outputs_EO.shape[0], -1)
                    outputs = C(outputs_EO)

                    _, predicted = torch.max(outputs.data, 1)
                    # total += labels.size(0)
                    # correct += (predicted == labels).sum().item()

                    outputs = torch.nn.functional.softmax(outputs, dim=-1)
                    # print(outputs)
                    # print(ii)
                    if ii == 0:
                        pred_outputs = outputs.cpu().detach()
                        # labels_all = labels.cpu().detach()
                        # name_all = name.cpu().detach()
                    else:
                        pred_outputs = torch.cat((pred_outputs, outputs.cpu().detach()), 0)
                        # labels_all = torch.cat((labels_all, labels.cpu().detach()), 0)
                        # name_all = torch.cat((name_all, name.cpu().detach()), 0)

                    ii += 1
                    print((ii - 1) * args.batch_size + inputs_SAR.shape[0])

                # print('Accuracy of the network on the ' + str(total) + ' test images: %d %%' % (100 * correct / total))
                # acc = 100 * correct / total
                # print('The acc of model {} is {}'.format(str(iter), acc))
                # print(pred_outputs.shape)
                all_pred_outputs[iter, :, :] = pred_outputs  # [10, 770, 10]

        all_pred_outputs = torch.squeeze(torch.mean(all_pred_outputs, 0))  # [770, 10]
        max_outputs, argmax_preds = torch.max(all_pred_outputs, 1)
        max_outputs = torch.squeeze(max_outputs)  # [770]
        preds1 = torch.squeeze(argmax_preds)  # [770]

        selected_list_ori = np.where(max_outputs.numpy() > 0.9)[0].tolist()
        selected_max_outputs = max_outputs[selected_list_ori]
        sorted_maxprob, indices_maxprob = torch.sort(selected_max_outputs, descending=True)  # from the larger to smaller
        selected_indices = indices_maxprob[:max(200, int(0.5 * len(indices_maxprob)))]
        selected_list = np.array(selected_list_ori)[selected_indices].tolist()

        rest_list_ = list(set(rest_list) - set(selected_list))
        rest_list = rest_list_

        test_data_train = test_data_


if __name__ == "__main__":
    args = parse_args()
    train()

