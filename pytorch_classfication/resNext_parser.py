from __future__ import division, absolute_import
import warnings
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

import time
import copy
import argparse
torch.manual_seed(0)

def make_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "parser", default="parser"
    # )
    # parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("--size_h", type=int, default='256', help="transforms.Resize((256,256))")
    parser.add_argument("--size_w", type=int, default='128', help="transforms.Resize((256,256))")
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--loss", type=str, default='cross')
    parser.add_argument("--w1_path", default='/Users/injungyou/Desktop/1/pair_nopadding_256*256.pt')
    parser.add_argument("--w2_path", default='/Users/injungyou/Desktop/1/pair_nopadding_256*256.pt')
    parser.add_argument("--w3_path", default='/Users/injungyou/Desktop/1/pair_nopadding_256*256.pt')
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
     transforms.ColorJitter(brightness=(0.6, 1.2), saturation=(0.5), hue=(-0.1, 0.1)),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
     ])
val_transforms = transforms.Compose(
    [transforms.Resize((args.size_h,args.size_w)), transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# define an object of the custom dataset for the train folder
# 데이터셋 만들기
data_dir = '/DATA/source/ij/pytorch_classfication/new_hat_datasets'
train_dataset = hat_Dataset(data_dir, train_transforms, 'train')
val_dataset = hat_Dataset(data_dir, val_transforms, 'val')
# 데이터 로더 만들기
# num_workers 숫자 높여서 학습하기 프로세스 여러개 띄어서 학습하기
train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
#
#modeling
class BottleNeck(nn.Module):
    expansion = 4
    Cardinality = 32 # group 수
    Basewidth = 64 # bottleneck 채널이 64이면 group convolution의 채널은 depth가 됩니다.
    Depth = 4 # basewidth일 때, group convolution의 채널 수
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        C = BottleNeck.Cardinality
        D = int(BottleNeck.Depth * out_channels / BottleNeck.Basewidth)

        self.conv_residual = nn.Sequential(
            nn.Conv2d(in_channels, C * D, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C*D),
            nn.ReLU(),
            nn.Conv2d(C*D, C*D, 3, stride=stride, padding=1, groups=BottleNeck.Cardinality, bias=False),
            nn.BatchNorm2d(C*D),
            nn.ReLU(),
            nn.Conv2d(C*D, out_channels * BottleNeck.expansion, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion)
        )

        self.conv_shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, 1, stride=stride, padding=0)

    def forward(self, x):
        x = self.conv_residual(x) + self.conv_shortcut(x)
        return x


# ResNext
class ResNext(nn.Module):
    def __init__(self, nblocks, num_classes=3, init_weights=True):
        super().__init__()
        self.init_weights=init_weights
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.conv2 = self._make_res_block(nblocks[0], 64, 1)
        self.conv3 = self._make_res_block(nblocks[1], 128, 2)
        self.conv4 = self._make_res_block(nblocks[2], 256, 2)
        self.conv5 = self._make_res_block(nblocks[3], 512, 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512 * BottleNeck.expansion, num_classes)

        # weights initialization
        if self.init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def _make_res_block(self, nblock, out_channels, stride):
        strides = [stride] + [1] * (nblock-1)
        res_block = nn.Sequential()
        for i, stride in enumerate(strides):
            res_block.add_module('dens_layer_{}'.format(i), BottleNeck(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * BottleNeck.expansion
        return res_block

    # weights initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def ResNext50():
    return ResNext([3, 4, 6, 3])


# check model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# x = torch.randn((3, 3, 256,256)).to(device)
model = ResNext50().to(device)

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


opt = optim.Adam(model.parameters(), lr=0.01)

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

        if epoch == num_epochs:
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


# exit()
######################<inference>##################################
# class hat_test_Dataset(Dataset):
#     def __init__(self, data_dir, transform, data_type='train'):
#         # path to images
#         path2data = os.path.join(data_dir, data_type)
#
#         # get a list of images
#         self.filenames = os.listdir(path2data)
#
#         # get the full path to images
#         self.full_filenames = [os.path.join(path2data, f) for f in self.filenames]
#         self.transform = transform
#
#     def __len__(self):
#         # return size of dataset
#         return len(self.full_filenames)
#
#     def __getitem__(self, idx):
#         # open image, apply transforms and return with label
#         image = Image.open(self.full_filenames[idx])
#         #         print('전',image)
#         image = self.transform(image)
#         #         print('후',image)
#         return image, self.filenames[idx]
#
#
# test_transformer = transforms.Compose(
#     [transforms.Resize((256, 256)), transforms.ToTensor(),
#      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
# # 데이터셋 만들기
# data_dir = '/DATA/source/ij/pytorch_hat_train_04/datasets/'
# test_dataset = hat_test_Dataset(data_dir, test_transformer, 'test_cctv')
#
# # 데이터 로더 만들기
# # num_workers 숫자 높여서 학습하기 프로세스 여러개 띄어서 학습하기
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=2)
#
# path2weights = '/DATA/source/ij/pytorch_hat_train_04/weights/0317_nopadding_256*256.pt'
#
# m = ResNext50().to(device)
# m.load_state_dict(torch.load(path2weights))
# m.eval()
#
# with torch.no_grad():
#     running_metric = 0.0
#     len_data = len(test_loader.dataset)
#     for xb, filename in test_loader:
#         count = 0
#         xb = xb.to(device)
#         print(xb.shape)
#         output = m(xb)
#         pred = output.argmax(1, keepdim=True)
#         pred = pred.cpu().numpy()
#         pred = pred.reshape(-1)
#         pred = pred.tolist()
#         filename = list(filename)
#         for i in range(len(pred)):
#             #             print(pred[i], filename[i][0])
#             if pred[i] == int(filename[i][0]):
#                 #             if pred[i] == 1 :
#                 count += 1
#         print(count / len(pred))

    #######################<test acc 확인>##################################
# test_transformer = transforms.Compose(
#     [transforms.Resize((256,256)), transforms.ToTensor(),transforms.Normalize((0.25, 0.25, 0.25), (0.25, 0.25, 0.25))])

# #데이터셋 만들기
# data_dir = '/DATA/source/ij/pytorch_classfication/datasets/'
# test_dataset = hat_Dataset(data_dir, test_transformer, 'test123')

# #데이터 로더 만들기
# #num_workers 숫자 높여서 학습하기 프로세스 여러개 띄어서 학습하기
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=2)
# print(test_dataset[0])

# path2weights = '/DATA/source/ij/pytorch_classfication/weights/weights_best_acc_256*256.pt'
# # 가중치 불러오기
# m = ResNext50().to(device)
# m.load_state_dict(torch.load(path2weights))
# m.eval()
# with torch.no_grad():
#     running_metric = 0.0
#     len_data = len(test_loader.dataset)
#     for xb, yb in test_loader:
#         xb = xb.to(device)
#         yb = yb.to(device)
#         output = m(xb)

#         loss_b, metric_b = loss_batch(nn.CrossEntropyLoss(reduction='sum'), output, yb, None)
#         print('전체수',output.shape[0],'metric_b(맞은것)',metric_b)

#         if metric_b is not None:
#             running_metric += metric_b


# metric = running_metric / len_data
# print('metrci',metric,len_data)


