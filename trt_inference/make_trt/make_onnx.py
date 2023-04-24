import sys
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
#############################추가####################################  
    def __init__(self):
        self.exp = get_exp(None, "yolox-m")
        self.preproc = ValTransform(legacy=False)
        self.cls_names = COCO_CLASSES
        self.num_classes = self.exp.num_classes
        self.confthre = self.exp.test_conf =0.5
        self.nmsthre = self.exp.nmsthre = 0.3
        self.test_size = self.exp.test_size = (640,640)
        self.cls_conf = 0.35
        self.conf = 0.5


    #모델 load하는 함수 
    def load(self):
        self.model = self.exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(self.model, self.exp.test_size)))
        self.model.cuda()
        self.model.eval()
        ckpt = torch.load("/DATA_17/ij/trt_inference/make_trt/yolox_m.pth", map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
            
if __name__ == '__main__':    
    max_batch_size = 1
    input = torch.randn(1,3, 640, 640).cuda()


    yd = YoloDetector()
    yd.load()
    model = yd.model
    output = model(input)

    # pytorch to onnx 
    torch.onnx.export(
        model,                                # model being run
        input,    # model input (or a tuple for multiple inputs)
        "yolox_m_14.onnx", # where to save the model (can be a file or file-like object)
        verbose=True, 
        opset_version=14,
        input_names = ['input'],              # the model's input names
        output_names = ['output'],
        dynamic_axes = {'input' : {0 : 'batch_size'},
                        'output' : {0 : 'batch_size'}}
    )            # the model's output names
