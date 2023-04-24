import sys
import time
import torch
from torch.backends import cudnn
# import cv2
from PIL import Image
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import transforms
import torchvision.transforms as transforms
import os

import torchvision.transforms.functional as TF

DEVICE = torch.device('cuda')



path2weights = '/DATA_17/ij/trt_inference/make_trt/helme_vit_2.pth'

batch_size = 1

input = torch.randn(batch_size,3, 224,224).cuda()  #배치, 채널, 세로(h), 가로(w)

model = torch.load(path2weights).to(DEVICE)
model.eval()

output = model(input)
print(output)



## pytorch to onnx 
torch.onnx.export(
    model,                                # model being run
    input,    # model input (or a tuple for multiple inputs)
    "helmet_vit5.onnx", # where to save the model (can be a file or file-like object)
    verbose=True, 
    opset_version=10,
#     do_constant_folding=True,
    input_names = ['input'],              # the model's input names
    output_names = ['output'],
    dynamic_axes = {'input' : {0 : 'batch_size'},
                    'output' : {0 : 'batch_size'}}
)            # the model's output names
