# -*- coding: utf8 -*-

from __future__ import division, absolute_import
import warnings
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import cv2 as cv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import random
import time
import copy
import argparse
torch.manual_seed(0)

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size_h", type=int, default=256)
    parser.add_argument("--size_w", type=int, default=128)
    parser.add_argument("--rotate", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_factor", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--loss", type=str, default='cross')
    parser.add_argument("--padding", type=str, default='no')
    parser.add_argument("--shift", type=str, default='no')
    parser.add_argument("--w1_path", default='/DATA/source/ij/pytorch_classfication/inception_weights/1.pt')
    parser.add_argument("--w2_path", default='/DATA/source/ij/pytorch_classfication/inception_weights/2.pt')
    parser.add_argument("--w3_path", default='/DATA/source/ij/pytorch_classfication/inception_weights/3.pt')
    
    return parser

args = make_parser().parse_args()

class hat_Dataset(Dataset):
    def __init__(self, data_dir, transform, data_type='train'):
        # path to images
        self.label_0 = 0
        self.label_1 = 0
        path2data = os.path.join(data_dir, data_type)

        # get a list of images
        filenames = os.listdir(path2data)

        # get the full path to images
        self.full_filenames = [os.path.join(path2data, f) for f in filenames]

        # labels are in a csv file named train_labels.csv
        csv_filename = data_type + '_labels.csv'
        path2csvLabels = os.path.join(data_dir, csv_filename)
        labels_df = pd.read_csv(path2csvLabels)

        # set data frame index to id
        labels_df.set_index('id', inplace=True)

        # obtain labels from data frame
        self.labels = [labels_df.loc[filename[:-4]].values[0] for filename in filenames]
        pp = self.labels
        if data_type == 'train':
            for i in pp:
                if i == 0:
                    self.label_0 += 1
                if i == 1:
                    self.label_1 += 1

        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.full_filenames)

    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        image = Image.open(self.full_filenames[idx])
        image = self.transform(image)
        return image, self.labels[idx]


train_transforms = transforms.Compose(
    [transforms.Resize((args.size_h,args.size_w)),
    #      transforms.RandomAffine(0.2),
    transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomRotation(degrees=(-args.rotate,args.rotate),fill = (int(0.485255), int(0.465255), int(0.485*255))),
    transforms.ColorJitter(brightness=(0.6, 1.2), saturation=(0.5), hue=(-0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
val_transforms = transforms.Compose(
    [transforms.Resize((args.size_h,args.size_w)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# define an object of the custom dataset for the train folder
data_dir = '/DATA/source/ij/pytorch_classfication/new_hat_datasets'
train_dataset = hat_Dataset(data_dir, train_transforms, 'train')
val_dataset = hat_Dataset(data_dir, val_transforms, 'val')
# 데이터 로더 만들기
# num_workers 숫자 높여서 학습하기 프로세스 여러개 띄어서 학습하기
train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
#


##########
# Basic layers
##########

import timm
inceptionV4 = timm.create_model('inception_v4', pretrained=False, num_classes=3)

#########
# Instantiation
##########

# check model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# x = torch.randn((3, 3, 256,256)).to(device)
model = inceptionV4.to(device)

# summary(model, (3, 256, 256), device=device.type)

# 학습하기
# define loss function, optimizer, lr_scheduler
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        print('hhhh')
    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

if args.loss == 'focal' :
    loss_func = FocalLoss()
elif args.loss == 'cross' :
    loss_func = nn.CrossEntropyLoss(reduction='mean')
elif args.loss == 'weight_loss' :
    label_0 = train_dataset.label_0
    label_1 = train_dataset.label_1
    label_max = max(label_0, label_1)
    ratio_label0 = label_max / label_0
    ratio_label1 = label_max / label_1
    print(ratio_label0, ratio_label1)  # 1.0 3.661923733636881
    label_tensor = torch.FloatTensor([ratio_label0, ratio_label1])
    label_tensor = label_tensor.to(device)
    loss_func = nn.CrossEntropyLoss(reduction='mean',weight = label_tensor)


# opt = optim.Adam(model.parameters(), lr=0.01)
opt = optim.Adam(model.parameters(), args.lr)

from torch.optim.lr_scheduler import ReduceLROnPlateau

lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=args.lr_factor, patience=args.patience)


# get current lr
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


# calculate the metric per mini-batch
def metric_batch(output, target):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects


# calculate the loss per mini-batch
def loss_batch(loss_func, output, target, opt=None):
    loss_b = loss_func(output, target)
    metric_b = metric_batch(output, target)

    if opt is not None:
        opt.zero_grad()
        loss_b.backward()
        opt.step()

    return loss_b.item(), metric_b


# calculate the loss per epochs
def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)
    print(len_data)

    for xb, yb in dataset_dl:

        xb = xb.to(device)
        yb = yb.to(device)

        output = model(xb)

        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)

        running_loss += loss_b

        if metric_b is not None:
            running_metric += metric_b

        if sanity_check is True:
            break

    loss = running_loss / len(dataset_dl)
    metric = running_metric / len_data
    return loss, metric


# function to start training
def train_val(model, params):
    num_epochs = params['num_epochs']
    loss_func = params['loss_func']
    opt = params['optimizer']
    train_dl = params['train_dl']
    val_dl = params['val_dl']
    sanity_check = params['sanity_check']
    lr_scheduler = params['lr_scheduler']
    path2weights = params['path2weights']
    path2weights2 = params['path2weights2']
    path2weights3 = params['path2weights3']
    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}
    acc = 0.0

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    start_time = time.time()

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)

        print('Epoch {}/{}, current lr= {}'.format(epoch, num_epochs - 1, current_lr))

        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, sanity_check, opt)
        loss_history['train'].append(train_loss)
        metric_history['train'].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, sanity_check)
        loss_history['val'].append(val_loss)
        metric_history['val'].append(val_metric)

        if acc < val_metric:
            acc = val_metric
            torch.save(model.state_dict(), path2weights2)
            print('best acc model weights!', acc)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            print('Copied best model weights!')

        if epoch == num_epochs -1 :
            torch.save(model.state_dict(), path2weights3)
            print('last weights!')

        lr_scheduler.step(train_loss)
        if current_lr != get_lr(opt):
            print('Loading best model weights!')
        #             model.load_state_dict(best_model_wts)

        print('train loss: %.6f, val loss: %.6f, accuracy: %.2f, time: %.4f min' % (
            train_loss, val_loss, 100 * val_metric, (time.time() - start_time) / 60))
        print('-' * 10)

    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history


# define the training parameters
params_train = {
    'num_epochs': args.epochs,
    'optimizer': opt,
    'loss_func': loss_func,
    'train_dl': train_loader,
    'val_dl': val_loader,
    'sanity_check': False,
    'lr_scheduler': lr_scheduler,
    'path2weights': args.w1_path,
    'path2weights2': args.w2_path,
    'path2weights3': args.w3_path,
}

# #######################<train>####################################
model, loss_hist, metric_hist = train_val(model, params_train)
