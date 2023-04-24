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
    parser.add_argument("--size_h", type=int, default=70)
    parser.add_argument("--size_w", type=int, default=70)
    parser.add_argument("--model", type=str, default='os',help ='resNext')
    parser.add_argument("--test", type=str, default='test_crop') #테스트 폴더명 
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--loss", type=str, default='cross')
    parser.add_argument("--padding", type=str, default='no')
    parser.add_argument("--w_path", default='/DATA/source/ij/pytorch_classfication/weights_osNet/f1_acc.pt')
    return parser

args = make_parser().parse_args()


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
    def __init__(self, nblocks, num_classes=2, init_weights=True):
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




# check model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# x = torch.randn((3, 3, 256,256)).to(device)



    
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
#         return image, self.labels[idx]


test_transformer = transforms.Compose(
    [transforms.Resize((args.size_h,args.size_w)), transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
#데이터셋 만들기 

data_dir = '/DATA/source/ij/pytorch_hat_train_04/datasets/'
test_dataset = hat_test_Dataset(data_dir, test_transformer, args.test)

#데이터 로더 만들기 
#num_workers 숫자 높여서 학습하기 프로세스 여러개 띄어서 학습하기
test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=True, num_workers=2)

path2weights = args.w_path

# if args.model == 'resNext' :
#     model = ResNext50().to(device)

if args.model == 'os' :
    model = osnet_x1_0().to(device)
if args.model == 'res' :
    model = ResNext50().to(device)

model.load_state_dict(torch.load(path2weights))
model.eval()

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
    if args.test == 'test_crop' :
        for xb, filename in test_loader:
            xb = xb.to(device)
            output = model(xb)
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
#                 print(pred[i],filename[i],filename[i][0])
    #         precision = count/len(pred)
        pre_0 = count_5/count_2
        pre_1 = count_4/count_3
        recall_0 = count_5/count_0
        recall_1 = count_4/count_1
        f1_score_0 = ((pre_0*recall_0)/(pre_0+recall_0))*2
        f1_score_1 = ((pre_1*recall_1)/(pre_1+recall_1))*2
        print('정답',(count_5 +count_4)/len_data)
        print('f1_score 0 :',f1_score_0)
        print('f1_score 1 :',f1_score_1)
        print('len_data',len_data)
#     else : 
#         with torch.no_grad():
#             running_metric = 0.0
#             len_data = len(test_loader.dataset)
#             for xb, yb in test_loader:
#                 xb = xb.to(device)
#                 yb = yb.to(device)
#                 output = model(xb)
#                 loss_b, metric_b = loss_batch(nn.CrossEntropyLoss(reduction='sum'), output, yb, None)
#                 pred = output.argmax(1, keepdim=True)
#                 pred = pred.cpu().numpy()
#                 yb = yb.cpu().numpy()
#                 pred = pred.reshape(-1)
#                 yb = yb.reshape(-1)
#                 pred = pred.tolist()
#                 yb = yb.tolist()
#                 filename = list(filename)
#                 for i in range(len(pred)):
#                     if int(yb[i]) == 0:
#                         count_0 += 1
#                     if int(yb[i]) == 1:
#                         count_1 += 1
#                     if pred[i] == 0 :
#                         count_2 += 1
#                     if pred[i] == 1 :
#                         count_3 += 1
#                     if (pred[i] == 0) and (int(yb[i]) == 0) :
#                         count_5 += 1
#                     if (pred[i] == 1) and (int(yb[i]) == 1) :
#                         count_4 += 1

#                 if metric_b is not None:
#                     running_metric += metric_b
#         pre_0 = count_5/count_2
#         pre_1 = count_4/count_3
#         recall_0 = count_5/count_0
#         recall_1 = count_4/count_1
#         f1_score_0 = ((pre_0*recall_0)/(pre_0+recall_0))*2
#         f1_score_1 = ((pre_1*recall_1)/(pre_1+recall_1))*2
#         print('f1_score 0 :',f1_score_0)
#         print('f1_score 1 :',f1_score_1)
#         metric = running_metric / len_data
#         print('metrci',metric,len_data)
