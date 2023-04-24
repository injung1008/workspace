import os
import sys
import numpy as np
import tensorrt as trt
import cv2 
import torchvision
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import time
import common


class Classifier:
    def __init__(self, logger):
        self.logger = logger
        self.device = torch.device('cuda:0')
        self.sm = nn.Softmax(dim=1)
        
        self.input_h = 256
        self.input_w = 128
        
    def load(self, weights):
        self.weights = weights
        self.Engine = common.Engine()      
        self.Engine.make_context(self.weights) 
#         self.init_sample_data()
        
    def init_sample_data(self):
        try:
            base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            sample_path = os.path.join(base_path, 'sample.png')
            sample_img = cv2.imread(sample_path)
            
            crops = [
                sample_img[213:588, 104:270, :],
                sample_img[75:288, 708:789, :],
                sample_img[181:439, 965:1065, :]
            ]
            self.inference_batch(crops)

        except Exception as e:
            self.logger.error(f'initialize sample image : {e}')
            self.logger.error(traceback.format_exc())

    
    def preprocess(self, imgs):
        result = torch.zeros([len(imgs), 3, self.input_h, self.input_w], dtype=torch.float32, device=torch.device("cuda:0")).fill_(144)

        for idx, img in enumerate(imgs):
            
#             img = img.permute(2,0,1).div(255)
            img = torch.from_numpy(img).to(torch.device("cuda")).permute(2, 0, 1).div(255)

            img = TF.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            
            img = TF.resize(img, (self.input_h, self.input_w)).float()
            
            result[idx, :, :self.input_h, :self.input_w] = img
        return result
    
    def postprocess(self,result):
        
        pred_scores = self.sm(result)
        pred = result.argmax(1,keepdim=True)
        return pred, pred_scores
    
    def inference(self,input_data) : 
        output_data = self.Engine.do_inference_v2(input_data) #결과 생성

        return output_data
    
    
    def inference_batch(self, img_batch):
        result_batch = list()
        t0 = time.time()
        img_batch = self.preprocess(img_batch)
        t1 = time.time()        
        result = self.inference(img_batch) #결과 생성
        t2 = time.time()
        preds, pred_scores = self.postprocess(result)
        t3 = time.time()
        
        for pred, pred_score in zip(preds, pred_scores):
            score = float(pred_score[pred])
            cls_pred = int(pred)
            result_batch.append((cls_pred, score))
            
        self.logger.info(f'psuit_trt 1.preprocess : {t1 - t0}, 2.inference : {t2 - t1}, 3.postprocess : {t3 - t2},  4.total : {t3 - t0}')

        return result_batch

def module_load(logger):
    cls = Classifier(logger)
    return cls     
    #SAMPLE 
if __name__ == '__main__':        
    # SAMPLE
    torch.cuda.init() #파이프 라인 필수 
    # trt_engine_path = f'/DATA_17/trt_test/engines/helmet3_test_ij/helmet3_int8_024.trt'
    trt_engine_path = f'/DATA_17/trt_test/engines/incep_helmet22_int8/incep_helmet_int8_064.trt'
    # #이미지 경로 설정     
    img_path = '/DATA_17/ij/non.jpg' 

    img = cv.imread(img_path, cv.IMREAD_COLOR)

    img_list = [img for _ in range(2)]

    helmet_cls = Classifier()
    #엔진로드
    helmet_cls.load(trt_engine_path)

    #이미지 preprocess & 스택 쌓기
    input_data = helmet_cls.preprocess(img_list)

    #inference
    output_data= helmet_cls.inference(input_data)

    pred, pred_scores = helmet_cls.postprocess(output_data)

    print(pred)
