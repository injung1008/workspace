from __future__ import division, absolute_import
import warnings
import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
# model
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# import package

# model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch import optim
import tqdm

# dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# display images
from torchvision import utils
# import matplotlib.pyplot as plt


# utils
import numpy as np
from torchsummary import summary
import time
import copy
torch.manual_seed(0)


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
    [transforms.Resize((256,256)),
#      transforms.RandomAffine(0.2),
     transforms.RandomRotation(0.2),
     transforms.ColorJitter(brightness=(0.6,1.2), saturation=(0.5), hue=(-0.1, 0.1)),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
     ])
val_transforms = transforms.Compose(
    [transforms.Resize((256,256)),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


# define an object of the custom dataset for the train folder
#데이터셋 만들기
data_dir = '/DATA/source/ij/pytorch_classfication/datasets/'
train_dataset = hat_Dataset(data_dir, train_transforms, 'train')
val_dataset = hat_Dataset(data_dir, val_transforms, 'val')
#데이터 로더 만들기
#num_workers 숫자 높여서 학습하기 프로세스 여러개 띄어서 학습하기
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

__all__ = [
    'osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25', 'osnet_ibn_x1_0'
]

pretrained_urls = {
    'osnet_x1_0':
        'https://drive.google.com/uc?id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY',
    'osnet_x0_75':
        'https://drive.google.com/uc?id=1uwA9fElHOk3ZogwbeY5GkLI6QPTX70Hq',
    'osnet_x0_5':
        'https://drive.google.com/uc?id=16DGLbZukvVYgINws8u8deSaOqjybZ83i',
    'osnet_x0_25':
        'https://drive.google.com/uc?id=1rb8UN5ZzPKRc_xvtHlyDh-cSz88YX9hs',
    'osnet_ibn_x1_0':
        'https://drive.google.com/uc?id=1sr90V6irlYYDd4_4ISU2iruoRG8J__6l'
}


##########
# Basic layers
##########
class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            groups=1,
            IN=False
    ):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups
        )
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=stride, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv3x3(nn.Module):
    """3x3 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution.
    1x1 (linear) + dw 3x3 (nonlinear).
    """

    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, padding=0, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            groups=out_channels
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


##########
# Building blocks for omni-scale feature learning
##########
class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(
            self,
            in_channels,
            num_gates=None,
            return_gates=False,
            gate_activation='sigmoid',
            reduction=16,
            layer_norm=False
    ):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            in_channels,
            in_channels // reduction,
            kernel_size=1,
            bias=True,
            padding=0
        )
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            in_channels // reduction,
            num_gates,
            kernel_size=1,
            bias=True,
            padding=0
        )
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError(
                "Unknown gate activation: {}".format(gate_activation)
            )

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x


class OSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(
            self,
            in_channels,
            out_channels,
            IN=False,
            bottleneck_reduction=4,
            **kwargs
    ):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return F.relu(out)


##########
# Network architecture
##########
class OSNet(nn.Module):
    """Omni-Scale Network.

    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. TPAMI, 2021.
    """

    def __init__(
            self,
            num_classes,
            blocks,
            layers,
            channels,
            feature_dim=512,
            loss='softmax',
            IN=False,
            **kwargs
    ):
        super(OSNet, self).__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1
        self.loss = loss
        self.feature_dim = feature_dim

        # convolutional backbone
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(
            blocks[0],
            layers[0],
            channels[0],
            channels[1],
            reduce_spatial_size=True,
            IN=IN
        )
        self.conv3 = self._make_layer(
            blocks[1],
            layers[1],
            channels[1],
            channels[2],
            reduce_spatial_size=True
        )
        self.conv4 = self._make_layer(
            blocks[2],
            layers[2],
            channels[2],
            channels[3],
            reduce_spatial_size=False
        )
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # fully connected layer
        self.fc = self._construct_fc_layer(
            self.feature_dim, channels[3], dropout_p=None
        )
        # identity classification layer
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self._init_params()

    def _make_layer(
            self,
            block,
            layer,
            in_channels,
            out_channels,
            reduce_spatial_size,
            IN=False
    ):
        layers = []

        layers.append(block(in_channels, out_channels, IN=IN))
        for i in range(1, layer):
            layers.append(block(out_channels, out_channels, IN=IN))

        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2)
                )
            )

        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None

        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x, return_featuremaps=False):
        x = self.featuremaps(x)
        if return_featuremaps:
            return x
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
#         if not self.training:
#             return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


def init_pretrained_weights(model, key=''):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    import os
    import errno
    import gdown
    from collections import OrderedDict

    def _get_torch_home():
        ENV_TORCH_HOME = 'TORCH_HOME'
        ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
        DEFAULT_CACHE_DIR = '~/.cache'
        torch_home = os.path.expanduser(
            os.getenv(
                ENV_TORCH_HOME,
                os.path.join(
                    os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch'
                )
            )
        )
        return torch_home

    torch_home = _get_torch_home()
    model_dir = os.path.join(torch_home, 'checkpoints')
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise
    filename = key + '_imagenet.pth'
    cached_file = os.path.join(model_dir, filename)

    if not os.path.exists(cached_file):
        gdown.download(pretrained_urls[key], cached_file, quiet=False)

    state_dict = torch.load(cached_file)
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights from "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(cached_file)
        )
    else:
        print(
            'Successfully loaded imagenet pretrained weights from "{}"'.
                format(cached_file)
        )
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                    format(discarded_layers)
            )


#########
# Instantiation
##########
def osnet_x1_0(num_classes=2, pretrained=False, loss='softmax', **kwargs):
    # standard size (width x1.0)
    model = OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        loss=loss,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, key='osnet_x1_0')
    return model

# check model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# x = torch.randn((3, 3, 256,256)).to(device)
model = osnet_x1_0().to(device)

# summary(model, (3, 256, 256), device=device.type)

# 학습하기
# define loss function, optimizer, lr_scheduler
loss_func = nn.CrossEntropyLoss(reduction='mean')
# loss_func = nn.CrossEntropyLoss(reduction='mean',weight = label_tensor)


opt = optim.Adam(model.parameters(), lr=0.01)

from torch.optim.lr_scheduler import ReduceLROnPlateau

lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)


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
            print('best acc model weights!',acc)

        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            print('Copied best model weights!')
        
        if epoch == num_epochs :
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
    'num_epochs': 1000,
    'optimizer': opt,
    'loss_func': loss_func,
    'train_dl': train_loader,
    'val_dl': val_loader,
    'sanity_check': False,
    'lr_scheduler': lr_scheduler,
    'path2weights':'/DATA/source/ij/pytorch_classfication/weights_osNet/osNet_256*256.pt',
    'path2weights2':'/DATA/source/ij/pytorch_classfication/weights_osNet/osNet_best_acc.pt',
    'path2weights3':'/DATA/source/ij/pytorch_classfication/weights_osNet/osNet_last_epoch_0317.pt',
}

# #######################<train>####################################
# model, loss_hist, metric_hist = train_val(model, params_train)


######################<inference>##################################
class hat_test_Dataset(Dataset):
    def __init__(self, data_dir, transform, data_type='train'):
        # path to images
        path2data = os.path.join(data_dir, data_type)

        # get a list of images
        self.filenames = os.listdir(path2data)

        # get the full path to images
        self.full_filenames = [os.path.join(path2data, f) for f in self.filenames]
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.full_filenames)

    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        image = Image.open(self.full_filenames[idx])
#         print('전',image)
        image = self.transform(image)
#         print('후',image)
        return image, self.filenames[idx]


test_transformer = transforms.Compose(
    [transforms.Resize((256,128)), transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
#데이터셋 만들기 
data_dir = '/DATA/source/ij/pytorch_hat_train_04/datasets/'
test_dataset = hat_test_Dataset(data_dir, test_transformer, 'test_cctv')

#데이터 로더 만들기 
#num_workers 숫자 높여서 학습하기 프로세스 여러개 띄어서 학습하기
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=2)

path2weights = '/DATA/source/ij/pytorch_classfication/weights_osNet/f1_acc.pt'

m = osnet_x1_0().to(device)
m.load_state_dict(torch.load(path2weights))
m.eval()
count = 0
count_0 = 0
count_1 = 0
count_2 = 0
count_3 = 0
count_4 = 0
count_5 = 0
with torch.no_grad():
    running_metric = 0.0
    len_data = len(test_loader.dataset)
    for xb, filename in test_loader:
        xb = xb.to(device)
        print(xb.shape)
        output = m(xb)
        pred = output.argmax(1, keepdim=True)
        pred = pred.cpu().numpy()
        pred = pred.reshape(-1)
        pred = pred.tolist()
        filename = list(filename)
        for i in range(len(pred)):
            if int(filename[i][0]) == 0:
                count_0 += 1
            if int(filename[i][0]) == 1:
                count_1 += 1
            if pred[i] == 0 :
                count_2 += 1
            if pred[i] == 1 :
                count_3 += 1
            if (pred[i] == 0) and (int(filename[i][0]) == 0) :
                count_5 += 1
            if (pred[i] == 1) and (int(filename[i][0]) == 1) :
                count_4 += 1
#         precision = count/len(pred)
    pre_0 = count_5/count_2
    pre_1 = count_4/count_3
    recall_0 = count_5/count_0
    recall_1 = count_4/count_1
    f1_score_0 = ((pre_0*recall_0)/(pre_0+recall_0))*2
    f1_score_1 = ((pre_1*recall_1)/(pre_1+recall_1))*2
    print('f1_score 0 :',f1_score_0)
    print('f1_score 1 :',f1_score_1)



    
    
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


