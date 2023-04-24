## torch to onnx
import sys
sys.path.insert(0,"/DATA_17/hjjo/YOLOX_pruning/YOLOX")



import os
import time
from loguru import logger

import cv2

import torch

# from yolox.data.data_augment import ValTransform
# from yolox.data.datasets import COCO_CLASSES
# from yolox.exp import get_exp
# from yolox.utils import fuse_model, get_model_info, postprocess, vis

path = './model_ij.pth'
model = torch.load(path)
model.cuda()
model.eval()
    
batch_size = 1
input = torch.randn(batch_size,3, 640, 640,requires_grad=True).cuda()
# yd = YoloDetector()
# yd.load()
# model = yd.model
output = model(input)

print(output)


## pytorch to onnx 
torch.onnx.export(
    model,                                # model being run
    input,    # model input (or a tuple for multiple inputs)
    "model_ij.onnx", # where to save the model (can be a file or file-like object)
    verbose=True, 
    opset_version=11,
    input_names = ['input'],              # the model's input names
    output_names = ['output'],
    dynamic_axes = {'input' : {0 : 'batch_size'},
                    'output' : {0 : 'batch_size'}}
)            # the model's output names



