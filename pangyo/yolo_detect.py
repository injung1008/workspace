#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import os
import time
from loguru import logger
import copy
import cv2
import numpy as np
import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis


class YoloDetector_park:
    def __init__(self):
        self.exp = get_exp(None, "yolox-x")
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
        ckpt = torch.load("yolox_x.pth", map_location="cpu")
#         # load the model state dict
        self.model.load_state_dict(ckpt["model"])
    
    #바운딩박스 그려주는 함수         
    def vis(self,img, boxes, scores,parking_id):
        for i in range(len(boxes)):
            box = boxes[i]
            score = scores[i]
            p_id = parking_id[i]
            if score < self.conf:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            color = (0, 0, 255)
            text = '{}:{:.1f}%'.format({p_id}, score * 100)
            txt_color = (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX

            txt_size = cv2.getTextSize(text, font, 0.6, 1)[0]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

            txt_bk_color = (0, 0, 0)
            cv2.rectangle(
                img,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
                txt_bk_color,
                -1
            )
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

        return img
    
    #사진 zone에 맞춰서 자르고, 결과 나오는 함수 
    def inference(self, frame_img, num_frame, zone, cut_rate_x, cut_rate_y):
        img_info = {"id": 0}
        img = frame_img
        img_info["file_name"] = num_frame
        zxs = [z for idx, z in enumerate(zone) if idx % 2 == 0]
        zys = [z for idx, z in enumerate(zone) if idx % 2 != 0]
        x1_min = min(zxs)
        x1_max = max(zxs)
        y1_min = min(zys)
        y1_max = max(zys)

        cut_w = round((x1_max - x1_min) * cut_rate_x)
        cut_h = round((y1_max - y1_min) * cut_rate_y)
        
        x_min = min(zxs) - cut_w
        x_max = max(zxs) + cut_w
        y_min = min(zys) - cut_h
        y_max = max(zys) + cut_h
        if x_min <0 :
            x_min = 0
        if x_max > img.shape[1] :
            x_max = img.shape[1] - 1
        if y_min <0 :
            y_min = 0
        if y_max > img.shape[0]:
            y_max = img.shape[0] - 1

        #사진 잘라주기 img0 = model에 input되는 잘린 image
        img0 = img[y_min:y_max, x_min:x_max]
        
        img_info["x_min"] = x_min
        img_info["y_min"] = y_min
        
        #잘린 이미지로 비율 만들어 주기 
        ratio = min(self.test_size[0] / img0.shape[0], self.test_size[1] / img0.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img0, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        img = img.cuda()

        outputs = self.model(img)
        outputs = postprocess(
            outputs, self.num_classes, self.confthre,
            self.nmsthre, class_agnostic=True
        )
 
        return outputs,img_info


class YoloDetector_st:
    def __init__(self):
        self.exp = get_exp(None, "yolox-x")
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
        ckpt = torch.load("yolox_x.pth", map_location="cpu")
#         # load the model state dict
        self.model.load_state_dict(ckpt["model"])
    
    
    #모델에서 결과값 도출 하기 
    def inference(self, frame_img, num_frame):
        img_info = {"id": 0}
        img = frame_img
        img_info["file_name"] = num_frame
        
        #이미지로 비율 만들어 주기 
        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio
          
        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        img = img.cuda()

   
        outputs = self.model(img)
        outputs = postprocess(
            outputs, self.num_classes, self.confthre,
            self.nmsthre, class_agnostic=True
        )
 
        return outputs,img_info

class YoloDetector_red_lights:
    def __init__(self):
        self.exp = get_exp(None, "yolox_x")
        self.preproc = ValTransform(legacy=False)
        self.cls_names = COCO_CLASSES
        self.num_classes = self.exp.num_classes
        self.confthre = self.exp.test_conf =0.5
        self.nmsthre = self.exp.nmsthre = 0.4
        self.test_size = self.exp.test_size = (640,640)
        self.cls_conf = 0.4
        self.conf = 0.5


    #모델 load하는 함수 
    def load(self):
        self.model = self.exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(self.model, self.exp.test_size)))
        self.model.cuda()
        self.model.eval()
        ckpt = torch.load("yolox_x.pth", map_location="cpu")
#         # load the model state dict
        self.model.load_state_dict(ckpt["model"])
    

    
    #사진 zone에 맞춰서 자르고, 결과 나오는 함수 
    def inference(self, frame_img, num_frame, zone, cut_rate_x, cut_rate_y,camera_number):
        img_info = {"id": 0}
        img = frame_img
        img_info["file_name"] = num_frame
        zxs = [z for idx, z in enumerate(zone) if idx % 2 == 0]
        zys = [z for idx, z in enumerate(zone) if idx % 2 != 0]
        x1_min = min(zxs)
        x1_max = max(zxs)
        y1_min = min(zys)
        y1_max = max(zys)

        cut_w = round((x1_max - x1_min) * cut_rate_x)
        cut_h = round((y1_max - y1_min) * cut_rate_y)
        
        x_min = min(zxs) - cut_w
        x_max = max(zxs) + cut_w
        y_min = min(zys) - cut_h
        y_max = max(zys) + cut_h
        if x_min <0 :
            x_min = 0
        if x_max > img.shape[1] :
            x_max = img.shape[1] - 1
        if y_min <0 :
            y_min = 0
        if y_max > img.shape[0]:
            y_max = img.shape[0] - 1
##########빨간불 설정 ( 화각변경시 좌표값 수정 ) #############
         
        #<시나리오 3>
        #빨간불 좌표 3082
        if camera_number == 3082 :
            if img[158][389][2] >= 100 or img[159][389][2] >= 100 or img[158][386][2] >= 100:
                img0 = img[y_min:y_max, x_min:x_max] 
                ratio = min(self.test_size[0] / img0.shape[0], self.test_size[1] / img0.shape[1])
                img, _ = self.preproc(img0, None, self.test_size)
            else:
                pts0 = np.array([[0, 0], [x_max, 0], [x_max, y_max], [0, y_max]], dtype=np.int32)
                cv2.fillConvexPoly(img, pts0, 1)
                ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
                img, _ = self.preproc(img, None, self.test_size)
        #빨간불 좌표 3044
        if camera_number == 3044 :
            if img[121][1316][2] >= 200 or img[122][1316][2] >= 200 or img[123][1316][2] >= 200:
                img0 = img[y_min:y_max, x_min:x_max]  
                ratio = min(self.test_size[0] / img0.shape[0], self.test_size[1] / img0.shape[1])
                img, _ = self.preproc(img0, None, self.test_size)
            else:
                pts0 = np.array([[0, 0], [x_max, 0], [x_max, y_max], [0, y_max]], dtype=np.int32)
                cv2.fillConvexPoly(img, pts0, 1)
                ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
                img, _ = self.preproc(img, None, self.test_size)
        #빨간불 좌표 3061
        if camera_number == 3061 :
            if img[93][1552][2] >= 200 or img[94][1554][2] >= 200:
                img0 = img[y_min:y_max, x_min:x_max]
                ratio = min(self.test_size[0] / img0.shape[0], self.test_size[1] / img0.shape[1])
                img, _ = self.preproc(img0, None, self.test_size)
            else:
                pts0 = np.array([[0, 0], [x_max, 0], [x_max, y_max], [0, y_max]], dtype=np.int32)
                cv2.fillConvexPoly(img, pts0, 1)
                ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
                img, _ = self.preproc(img, None, self.test_size)

        #빨간불 좌표 3065
        if camera_number == 3065 :
            if img[225][1750][2] >= 200 or img[223][1749][2] >= 200:
                img0 = img[y_min:y_max, x_min:x_max] 
                ratio = min(self.test_size[0] / img0.shape[0], self.test_size[1] / img0.shape[1])
                img, _ = self.preproc(img0, None, self.test_size)

            else:
                pts0 = np.array([[0, 0], [x_max, 0], [x_max, y_max], [0, y_max]], dtype=np.int32)
                cv2.fillConvexPoly(img, pts0, 1)
                ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
                img, _ = self.preproc(img, None, self.test_size)

        #<시나리오 4>
        if camera_number == 4 :
            img0 = img[y_min:y_max, x_min:x_max]
            ratio = min(self.test_size[0] / img0.shape[0], self.test_size[1] / img0.shape[1])
            img, _ = self.preproc(img0, None, self.test_size)
        
        img_info["x_min"] = x_min
        img_info["y_min"] = y_min
        
        #잘린 이미지로 비율 만들어 주기 
        
        img_info["ratio"] = ratio

        
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        img = img.cuda()

        outputs = self.model(img)
        outputs = postprocess(
            outputs, self.num_classes, self.confthre,
            self.nmsthre, class_agnostic=True
        )
 
        return outputs,img_info



class YoloDetector:
    def __init__(self):
        self.exp = get_exp(None, "yolox-x")
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
        ckpt = torch.load("yolox_x.pth", map_location="cpu")
#         # load the model state dict
        self.model.load_state_dict(ckpt["model"])
    
    
    #모델에서 결과값 도출 하기 
    def inference(self, frame_img):

        img_info = {"id": 0}
        img = frame_img

        #이미지로 비율 만들어 주기 
        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        img = img.cuda()

        outputs = self.model(img)
        outputs = postprocess(
            outputs, self.num_classes, self.confthre,
            self.nmsthre, class_agnostic=True
        )
 
        return outputs,img_info

