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
    parser.add_argument("--lr_factor", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--loss", type=str, default='cross')
    parser.add_argument("--padding", type=str, default='no')
    parser.add_argument("--shift", type=str, default='no')
    parser.add_argument("--w1_path", default='/DATA/source/ij/pytorch_classfication/1/1.pt')
    parser.add_argument("--w2_path", default='/DATA/source/ij/pytorch_classfication/1/2.pt')
    parser.add_argument("--w3_path", default='/DATA/source/ij/pytorch_classfication/1/3.pt')
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
#         # shift
#         if args.shift == 'shift':
#             img = np.array(image)
#             w_1 = round(int(img.shape[1] * 0.1))
#             h_1 = round(int(img.shape[0] * 0.1))
#             if w_1 == 0 or h_1 == 0 :
#                 pass
#             else :
#                 a = random.randrange(-w_1,w_1)
#                 b = random.randrange(-h_1,h_1)
#                 translation_matrix = np.float32([[1,0,a], [0,1,b]])
#                 img_translation = cv.warpAffine(img, translation_matrix, (w_1,h_1))
#                 image = Image.fromarray(img_translation)
#         else : pass

#         #패딩
#         if args.padding == 'padding':
#             img = np.array(image)
#             h_1 = img.shape[0]
#             w_1 = img.shape[1]
#             if h_1 <= w_1 * 2:
#                 img = cv.copyMakeBorder(img, 0, round((w_1 * 2) - h_1), 0, 0, cv.BORDER_CONSTANT,(0.485, 0.456, 0.406))
#             elif w_1 <= h_1/2:

#                 img = cv.copyMakeBorder(img, 0, 0, 0, round((h_1/2) - w_1), cv.BORDER_CONSTANT,(0.485, 0.456, 0.406))
#             image = Image.fromarray(img)
#         else : pass

        image = self.transform(image)
        return image, self.labels[idx]



train_transforms = transforms.Compose(
    [transforms.Resize((args.size_h,args.size_w)),
     #      transforms.RandomAffine(0.2),
     transforms.RandomHorizontalFlip(p=0.5),
     #transforms.RandomRotation(degrees=(-args.rotate,args.rotate),fill = (int(0.485*255), int(0.465*255), int(0.485*255))),
     transforms.ColorJitter(brightness=(0.6, 1.2), saturation=(0.5), hue=(-0.1, 0.1)),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
     ])
val_transforms = transforms.Compose(
    [transforms.Resize((args.size_h,args.size_w)), transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# define an object of the custom dataset for the train folder
# 데이터셋 만들기
data_dir = '/DATA/source/ij/pytorch_classfication/new_hat_datasets/'
train_dataset = hat_Dataset(data_dir, train_transforms, 'train')
val_dataset = hat_Dataset(data_dir, val_transforms, 'val')
# 데이터 로더 만들기
# num_workers 숫자 높여서 학습하기 프로세스 여러개 띄어서 학습하기
train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
#

__all__ = [
    'osnet_ain_x1_0', 'osnet_ain_x0_75', 'osnet_ain_x0_5', 'osnet_ain_x0_25'
]

pretrained_urls = {
    'osnet_ain_x1_0':
    'https://drive.google.com/uc?id=1-CaioD9NaqbHK_kzSMW8VE4_3KcsRjEo',
    'osnet_ain_x0_75':
    'https://drive.google.com/uc?id=1apy0hpsMypqstfencdH-jKIUEFOW4xoM',
    'osnet_ain_x0_5':
    'https://drive.google.com/uc?id=1KusKvEYyKGDTUBVRxRiz55G31wkihB6l',
    'osnet_ain_x0_25':
    'https://drive.google.com/uc?id=1SxQt2AvmEcgWNhaRb2xC4rP6ZwVDP0Wt'
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
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


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
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, stride=1, bn=True):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=stride, padding=0, bias=False
        )
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
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
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


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
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        return self.relu(x)


class LightConvStream(nn.Module):
    """Lightweight convolution stream."""

    def __init__(self, in_channels, out_channels, depth):
        super(LightConvStream, self).__init__()
        assert depth >= 1, 'depth must be equal to or larger than 1, but got {}'.format(
            depth
        )
        layers = []
        layers += [LightConv3x3(in_channels, out_channels)]
        for i in range(depth - 1):
            layers += [LightConv3x3(out_channels, out_channels)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


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
        self.relu = nn.ReLU()
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
            self.gate_activation = nn.ReLU()
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

    def __init__(self, in_channels, out_channels, reduction=4, T=4, **kwargs):
        super(OSBlock, self).__init__()
        assert T >= 1
        assert out_channels >= reduction and out_channels % reduction == 0
        mid_channels = out_channels // reduction

        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList()
        for t in range(1, T + 1):
            self.conv2 += [LightConvStream(mid_channels, mid_channels, t)]
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = 0
        for conv2_t in self.conv2:
            x2_t = conv2_t(x1)
            x2 = x2 + self.gate(x2_t)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        return F.relu(out)


class OSBlockINin(nn.Module):
    """Omni-scale feature learning block with instance normalization."""

    def __init__(self, in_channels, out_channels, reduction=4, T=4, **kwargs):
        super(OSBlockINin, self).__init__()
        assert T >= 1
        assert out_channels >= reduction and out_channels % reduction == 0
        mid_channels = out_channels // reduction

        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList()
        for t in range(1, T + 1):
            self.conv2 += [LightConvStream(mid_channels, mid_channels, t)]
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels, bn=False)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = 0
        for conv2_t in self.conv2:
            x2_t = conv2_t(x1)
            x2 = x2 + self.gate(x2_t)
        x3 = self.conv3(x2)
        x3 = self.IN(x3) # IN inside residual
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
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
        conv1_IN=False,
        **kwargs
    ):
        super(OSNet, self).__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1
        self.loss = loss
        self.feature_dim = feature_dim

        # convolutional backbone
        self.conv1 = ConvLayer(
            3, channels[0], 7, stride=2, padding=3, IN=conv1_IN
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(
            blocks[0], layers[0], channels[0], channels[1]
        )
        self.pool2 = nn.Sequential(
            Conv1x1(channels[1], channels[1]), nn.AvgPool2d(2, stride=2)
        )
        self.conv3 = self._make_layer(
            blocks[1], layers[1], channels[1], channels[2]
        )
        self.pool3 = nn.Sequential(
            Conv1x1(channels[2], channels[2]), nn.AvgPool2d(2, stride=2)
        )
        self.conv4 = self._make_layer(
            blocks[2], layers[2], channels[2], channels[3]
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

    def _make_layer(self, blocks, layer, in_channels, out_channels):
        layers = []
        layers += [blocks[0](in_channels, out_channels)]
        for i in range(1, len(blocks)):
            layers += [blocks[i](out_channels, out_channels)]
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
            layers.append(nn.ReLU())
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

            elif isinstance(m, nn.InstanceNorm2d):
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
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
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
        if not self.training:
            return v
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
            k = k[7:] # discard module.

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


##########
# Instantiation
##########
def osnet_x1_0(
    num_classes=3, pretrained=False, loss='softmax', **kwargs
):
    model = OSNet(
        num_classes,
        blocks=[
            [OSBlockINin, OSBlockINin], [OSBlock, OSBlockINin],
            [OSBlockINin, OSBlock]
        ],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        loss=loss,
        conv1_IN=True,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, key='osnet_ain_x1_0')
    return model



# check model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# x = torch.randn((3, 3, 256,256)).to(device)
model = osnet_x1_0().to(device)

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
