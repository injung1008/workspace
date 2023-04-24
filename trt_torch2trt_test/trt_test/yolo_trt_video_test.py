import time
import torch
from torch.backends import cudnn
from matplotlib import colors
import cv2 as cv
import numpy as np
import os
import statistics
from skimage import io, draw
from loguru import logger
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from yolox_detector import YOLOX_DETECTOR
from classification import CLASSIFICATION
import yolox_detector
import classification
import convert
import copy
import csv
torch.cuda.init()



def getFrameByVideo(video_path):
    frame_list = []
    
    cap = cv.VideoCapture(video_path)
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv.CAP_PROP_FPS)
    cnt = 0 
    if cap.isOpened() :    
        while True:         
            ret, frame = cap.read()
            if ret == False :
                break
#             if cnt < 1:
#                 cnt+=1
#                 continue
#             if cnt > 26:
#                 break
            frame_list.append(frame)
            cnt += 1

    print(f"{cnt},{len(frame_list)} frame loaded.")
    return frame_list, width, height, fps
    




#detector box값 도출 해주는것 
def visual(output, ratio, cls_conf=0.35):
    #아무런 output이 나오지 않을경우 
    if output is None:
        return None, None, None
    output = output.cpu()
    bboxes = output[:, 0:4]
    # preprocessing: resize
    bboxes /= ratio
    cls = output[:, 6] #현재 이미지에서 잡힌 객체들의 번호 차 = 2번 , 신호등 = 9 번 
    scores = output[:, 4] * output[:, 5] #잡힌 객체들의 신뢰도 수준 
    original_bboxes = []
    original_cls = []
    original_scores = []
    for i in range(len(bboxes)):
        box = bboxes[i]
        cls_id = int(cls[i])
        score = scores[i]
        
        x_min = int(box[0])
        y_min = int(box[1])
        x_max = int(box[2])
        y_max = int(box[3])
        
        if cls_conf < score :
            original_cls.append(cls_id)
            original_scores.append(score)
            original_bboxes.append([x_min,y_min,x_max,y_max])

    #output은 나왔지만 차량에 해당 하는 output이 없을경우 (사람만 포착이 되었을경우)     
    if len(original_bboxes) == 0 :
        return None, None, None

    return original_bboxes, original_cls, original_scores


    
class Model:
    def __init__(self):
#         self.trt_engine_path = '/DATA_17/engines/fp16/yoloxm_fp16_024.trt'
        self.trt_engine_path = '/DATA_17/trt_test/engines/yoloxm_work_s/yoloxm_fp16_024.trt'
        self.yd = YOLOX_DETECTOR()
    def initialize(self):
        self.yd.load(self.trt_engine_path)
 
    def proc(self,video_path,output_path):
        frame_list, width, height, fps = getFrameByVideo(video_path)
        stack = 24

        vid_writer = cv.VideoWriter(
            output_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )

        t0 = 0
        divide = round((len(frame_list)/stack))
        
        ##### 로딩하기 ######
        for i in range(0,50):
            input_data = frame_list[t0:t0+stack]
            t0 += stack
            outputs = self.yd.total_inference(input_data)
        ##### 로딩하기 ######
        
        ##### 인퍼런스 ######
        t = 0
        divide = round((len(frame_list)/stack))
        start = time.time()
        output_results = [] 
        
        for i in range(divide):
            input_data = frame_list[t:t+stack]
            t += stack
            
            infer_start_time = time.time()
            outputs = self.yd.total_inference(input_data)
            output_results.append(outputs)
            print(f"infer_total_time : {time.time() - infer_start_time}")
#             print(outputs)
        print(f"total_time : {time.time() - start}")

        ##### 인퍼런스 ######
        
        ##### 박스 그리기 & csv 만들기 ######
        p = 0
        with open('./output/hat_ij_trt.csv', 'w', encoding='UTF-8') as f:
            w = csv.writer(f)
            for i in range(divide):
                input_data = frame_list[p:p+stack]
                p += stack
                outputs = output_results[i]

                for frame, pred in zip(input_data, outputs) :
                    ratio = min(640 / frame.shape[0], 640 / frame.shape[1])        
                    bboxes, class_id, scores  = visual(pred, ratio, cls_conf=0.35)
                    data = f'{class_id}'.split(" ")
                    w.writerow(data)
                    if bboxes == None : 
                        result_frame = frame
                    else : 
                        result_frame = yolox_detector.vis(frame, bboxes, scores, class_id, conf=0.5, class_names=None)
                    vid_writer.write(result_frame)
                    ch = cv.waitKey(1)
                    if ch == 27 or ch == ord("q") or ch == ord("Q"):
                        break


    


input_list = [
    './input/hat_ij.mp4' 
]
output_list = [
    './output/hat_ij_trt.mp4',
]





m = Model()
m.initialize()
m.proc(input_list[0],output_list[0])

