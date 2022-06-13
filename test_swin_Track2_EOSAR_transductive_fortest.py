import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as data
import torchvision
from torchvision import transforms, models
import argparse
from torch.utils.data import DataLoader, Dataset
from skimage import io, transform
import numpy as np
import os
from PIL import Image
import random
from swin_transformer import SwinTransformer

import csv
import scipy.io as sio
import time


def parse_args():
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
    parser.add_argument('--batch_size', help=h['batch_size'], type=int, default=128)
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
        m.inplace = True


class MyData_test_new(Dataset):
    def __init__(self, root_dir_SAR, root_dir_EO, transform=None):
        self.root_dir_SAR = root_dir_SAR
        self.root_dir_EO = root_dir_EO
        self.transform = transform

        self.all_files = []
        for file in sorted(os.listdir(self.root_dir_SAR)):
            # print(os.path.join(root, file))
            if 'EO' not in file:
                self.all_files.append(self.root_dir_SAR + '/' + file)

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        img_path_SAR = self.all_files[index]
        img_path_EO = img_path_SAR.replace('SAR', 'EO')
        img_SAR = Image.open(img_path_SAR)
        img_EO = Image.open(img_path_EO)
        img_SAR = img_SAR.convert('RGB')
        img_EO = img_EO.convert('RGB')
        name = int(img_path_SAR.split('/')[-1].split('_')[-1].split('.')[0])   # read the number of the images

        if self.transform:
            img_SAR = self.transform(img_SAR)
            img_EO = self.transform(img_EO)

        return img_SAR, img_EO, index, name


def load_test_data(test_data_path_SAR, test_data_path_EO, batch_size):
    transform_test = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_data = MyData_test_new(test_data_path_SAR, test_data_path_EO, transform=transform_test)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    return test_data, test_data_loader


class Classifier(nn.Module):
    def __init__(self, num_class):
        super(Classifier, self).__init__()
        self.ln1 = nn.Linear(1024, 4096)
        self.ln2 = nn.Linear(4096, 4096)
        self.ln3 = nn.Linear(4096, num_class)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.ln1(x))
        x1 = self.relu(self.ln2(x1))
        x1 = self.ln3(x1)
        return x1


def test():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data_path_SAR = './test_images_SAR'
    test_data_path_EO = './test_images_EO'
    _, test_data_loader = load_test_data(test_data_path_SAR, test_data_path_EO, args.batch_size)

    F1 = SwinTransformer(img_size=224, patch_size=4, in_chans=2, num_classes=10,
                        embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                        window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                        # use_checkpoint=False
                        use_checkpoint=True
                        )  # the feature dim is 1024
    F1.apply(inplace_relu)
    F1 = torch.load('swinB_F_bestacc_Track2_EOSAR_iteration_num1_fortest.pth')
    F1 = F1.cuda()
    C1 = Classifier(10)
    C1 = torch.load('swinB_C_bestacc_Track2_EOSAR_iteration_num1_fortest.pth')
    C1.apply(inplace_relu)
    C1 = C1.cuda()
    F1.eval()
    C1.eval()

    start = time.time()

    with torch.no_grad():
        # all_pred_outputs = torch.zeros(10, 826, 10)
        all_pred_outputs = torch.zeros(1, 826, 10)
        # for iter in range(10):
        for iter in range(1):
            ii = 0
            for inputs_SAR, inputs_EO, _, name in test_data_loader:
                inputs_SAR, inputs_EO, name = inputs_SAR.to(device), inputs_EO.to(device), name.to(device)

                inputs = torch.cat((inputs_EO[:, 0, :, :].unsqueeze(1), inputs_SAR[:, 0, :, :].unsqueeze(1)), 1)

                outputs = F1(inputs)
                outputs = outputs.view(outputs.shape[0], -1)
                outputs = C1(outputs)
                _, predicted = torch.max(outputs.data, 1)
                outputs = torch.nn.functional.softmax(outputs, dim=-1)

                if ii == 0:
                    pred_outputs = outputs.cpu().detach()
                    name_all = name.cpu().detach()
                else:
                    pred_outputs = torch.cat((pred_outputs, outputs.cpu().detach()), 0)
                    name_all = torch.cat((name_all, name.cpu().detach()), 0)

                ii += 1

            all_pred_outputs[iter, :, :] = pred_outputs

    end = time.time()

    print('total testing time is:', end-start)
    print('time of each test image is:', (end-start) / 826)

    sio.savemat('EOSAR_iteration_num1_fortest_bestacc.mat', {'preds': all_pred_outputs.numpy()})

    all_pred_outputs = torch.squeeze(torch.mean(all_pred_outputs, 0))
    _, predicted = torch.max(all_pred_outputs, 1)

    csv_file = open('results_Track2_EOSAR_iteration_num1_fortest_bestacc.csv', 'w', newline='')
    fwriter = csv.writer(csv_file)
    row_now = ['image_id', 'class_id']
    fwriter.writerow(row_now)
    for i in range(826):
        row_now = [name_all[i].tolist(), predicted[i].tolist()]
        fwriter.writerow(row_now)
    csv_file.close()


if __name__ == "__main__":
    args = parse_args()
    test()

