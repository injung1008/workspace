## torch to onnx
import sys
sys.path.insert(0,"/DATA_17/hjjo/YOLOX_pruning/YOLOX")



import os
import time
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis


class YoloDetector:
    def __init__(self):
        self.exp = get_exp(None, "yolox-m")
        self.preproc = ValTransform(legacy=False)
        self.cls_names = COCO_CLASSES
        self.num_classes = self.exp.num_classes
        self.confthre = self.exp.test_conf
        self.nmsthre = self.exp.nmsthre
        self.test_size = self.exp.test_size
        self.cls_conf = 0.35
        print(self.cls_names)

    def load(self):
        self.model = self.exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(self.model, self.exp.test_size)))
        self.model.cuda()
        self.model.eval()
        
        ckpt = torch.load("yolox_m.pth", map_location="cpu")
#         # load the model state dict
        self.model.load_state_dict(ckpt["model"])

    
batch_size = 1
input = torch.randn(batch_size,3, 640, 640,requires_grad=True).cuda()
yd = YoloDetector()
yd.load()
model = yd.model
output = model(input)

print(output)


## pytorch to onnx 
torch.onnx.export(
    model,                                # model being run
    input,    # model input (or a tuple for multiple inputs)
    "yolox_m_model_fp16.onnx", # where to save the model (can be a file or file-like object)
    verbose=True, 
    opset_version=11,
    input_names = ['input'],              # the model's input names
    output_names = ['output'],
    dynamic_axes = {'input' : {0 : 'batch_size'},
                    'output' : {0 : 'batch_size'}}
)            # the model's output names