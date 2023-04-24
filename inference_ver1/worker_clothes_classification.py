import os
import sys
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import cv2 as cv
import torchvision
import torch
import torchvision.transforms.functional as TF
import time
import common


#이름 변경 trt 빼기 
class WORKER_CLASSIFICATION:
    def __init__(self):
        self.start_img_stack = 1

        
    def img_process(self,img):
        img  = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = torch.from_numpy(img).cuda()
        img = img.permute(2, 0, 1)
        img = TF.resize(img,(self.input_h, self.input_w))
        img = img.div(255)
        img = TF.normalize(img,(0.485, 0.456, 0.406), (0.229, 0.224, 0.225))         
        img_list = [img for _ in range(self.start_img_stack)]
        imgs = torch.stack(img_list,dim=0)
        return imgs


    def make_output(self,result):
        result = torch.from_numpy(result)
        result = result[0:self.img_batch*3]
        result = result.reshape(self.img_batch,-1) 
        pred = result.argmax(1,keepdim=True)
        return pred

    def load(self,trt_engine_path):
        #Engine class 소환 
        self.Engine = common.Engine()      
        #inference에 필요한 context 만들어주기  
        self.ctx = self.Engine.make_context(trt_engine_path)
        #버퍼 할당해주기 
        self.max_batch, self.input_w, self.input_h = self.Engine.allocate_buffers_all()


    def inference(self,input_data) : 
        result, self.img_batch = self.Engine.do_inference_v2(input_data) #결과 생성

        return result

    def release(self):
        self.Engine.flush()


# #SAMPLE
# trt_engine_path = f'/DATA_17/ij/trt_inference/make_trt/worker2.trt'

# # #이미지 경로 설정     
# img_path = '/DATA_17/ij/non.jpg' 

# img = cv.imread(img_path, cv.IMREAD_COLOR)


# worker_cls = WORKER_CLASSIFICATION()
# #엔진로드
# worker_cls.load(trt_engine_path)

# #이미지 preprocess & 스택 쌓기
# img_stack = worker_cls.img_process(img)

# #input 버퍼할당해 주지 않고 데이터를 바로 보낼때 (선택적 옵션) -> inference에서 자동으로 np인지 tensor인지 구분해서 inference
# input_data = torch.tensor(img_stack).cuda() 

# #inference
# result = worker_cls.inference(input_data)

# #postprocess
# outputs = worker_cls.make_output(result)

# print(outputs)






