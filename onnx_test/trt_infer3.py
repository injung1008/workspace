import tensorrt as trt
import numpy as np
import os

import pycuda.driver as cuda
import pycuda.autoinit
import cv2 as cv

import argparse
import os
import time
from loguru import logger
import numpy as np
import cv2

import torch


import sys

sys.path.insert(0,"/DATA_17/hjjo/YOLOX_pruning/YOLOX")

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis





class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TrtModel:
    
    def __init__(self,engine_path,max_batch_size=1,dtype=np.float32):
        
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        print('self.logger',self.logger)
        self.runtime = trt.Runtime(self.logger)
        print('self.runtime',self.runtime)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        
        print(f"self.engine : {self.engine.get_binding_shape(0)}")

        self.max_batch_size = max_batch_size
        print(f"self.max_batch_size : {self.max_batch_size}")

        
        
        self.context = self.engine.create_execution_context()
        print(f"self.context.get_binding_shape(0) : {self.context.get_binding_shape(0)}")

        self.context.set_binding_shape(0, (max_batch_size, 3, 224, 224))
        print(f"self.context.get_binding_shape(0) : {self.context.get_binding_shape(0)}")
        
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        print(f"inputs len : {len(self.inputs)}")


                
                
    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
#             print(engine_data)
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        print('engine',engine)
        return engine
    
    def allocate_buffers(self):
        
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
#         print(self.engine.get_binding_dtype(0))
        print(self.engine.get_binding_shape("input"))
        print(self.engine.get_binding_shape("output"))
        for binding in self.engine: 
            print(f"binding : {binding}")
            
            if self.engine.binding_is_input(binding):
                size = trt.volume((1,3,224,224)) * self.max_batch_size
            else:
                size = trt.volume((1,2048)) * self.max_batch_size
                
                
#             size = trt.volume((1,3,640,640)) * self.max_batch_size
            print(f"size : {size}")
#             size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
       
            
    def __call__(self,x:np.ndarray,batch_size=1):
        x = x.astype(self.dtype)
        print(f"x.ravel() shape : {x.ravel().shape}")

#         np.copyto(self.inputs.host,x.ravel())
        np.copyto(self.inputs[0].host,x.ravel())
        start = time.time()
        
        
# #######        
#         # Transfer input data to the GPU.
#         [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
#         # Run inference.
#         context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
#         # Transfer predictions back from the GPU.
#         [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
#         # Synchronize the stream
# ########        
        
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
#             print(f"inp.host.shape : {inp.host.shape}")
#             print(f"inp.host : {inp.host}")
        
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
#         self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
#             print('self.outputs.shape',out.host.reshape(batch_size,-1))
#             print(f"out.host.shape : {out.host.shape}")
#             print(f"out.host : {out.host}")
            

        
        self.stream.synchronize()
        
        end = time.time()
        
        print(f"infer time : {end - start}")

        cnt = 0
        for out in self.outputs :
            cnt += 1
            out.host.reshape(batch_size,-1) 
#         print('cnt',cnt)
        return [out.host.reshape(batch_size,-1) for out in self.outputs]

# trt_engine_path = '/DATA_17/ij/yolox_s_model.trt'
# model = TrtModel(trt_engine_path)
    

    
def visual(output, ratio,cls_conf=0.5):


    #아무런 output이 나오지 않을경우 
    if output is None:
        return None, None, None
    output = output.cpu()
    bboxes = output[:, 0:4]
    # preprocessing: resize
    bboxes /= ratio
    cls = output[:, 6] 
    scores = output[:, 4] * output[:, 5] 
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

def drawBoundingBox(img,original_bboxes,original_cls):
    for i in range(len(original_bboxes)):
        box = original_bboxes[i]
        x1,y1,x2,y2 = (box[0],box[1],box[2],box[3])
        cls = original_cls[i]
        label = str(cls)


        cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,255))
        cv2.putText(img,label,(x2-20,y2+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)

    return ori_img

def preproc(img_path):
    preproc = ValTransform(legacy=False)   
    img = cv.imread(img_path, cv.IMREAD_COLOR)
    img, _ = preproc(img, None, (224,224))
    return img
    
def make_output(result,exp):
    outputs = torch.Tensor(result)
    outputs = outputs.view([-1, 8400,85])    
    
    cls_names = COCO_CLASSES
    num_classes = exp.num_classes
    confthre = exp.test_conf =0.5
    nmsthre = exp.nmsthre =0.3

    outputs = postprocess(
        outputs, num_classes, confthre,
        nmsthre, class_agnostic=True)

    return outputs





if __name__ == "__main__":
 
    batch_size = 128
#     trt_engine_path = '/DATA_17/ij/onnx_model/resnet50_b124.trt'
    trt_engine_path = '/DATA_17/ij/onnx_model/resnet50_2_4_s_b124.trt'
    
    img_path = '/DATA_17/hjjo/YOLOX_pruning/YOLOX/test_image.jpeg'
    exp = get_exp("/DATA_17/hjjo/YOLOX_pruning/YOLOX/exps/default/yolox_m.py",None)
    
    model = TrtModel(trt_engine_path,max_batch_size=batch_size,dtype=np.float32)
#     shape = model.engine.get_binding_shape(0)
#     print(shape)
#     inputs, outputs, bindings, stream = model.allocate_buffers(batch_size)
    #이미지 받아서 읽어오고 사이즈 변환 해주기
    img = preproc(img_path)
#     img = img.astype(np.float16)
    #결과값 뽑아오기 
    
    img_list = [img for _ in range(batch_size)] 
    img_stack = np.stack(img_list, axis=0)
    print(f"img_stack len : {len(img_stack)}")

    for _ in range(10):
        result = model(img_stack,batch_size)
#         result1 = np.reshape(result,(batch_size,1,714000))
    #결과값 텐서 변환 등 해주기
#     outputs = make_output(result1,exp)
#     print(outputs)

        
#     ratio = min(640 / img.shape[0], 640 / img.shape[1])
#     #박스, 클래스 도출 
#     original_bboxes, original_cls, original_scores = visual(outputs[0], ratio,cls_conf=0.5)
    
#     ori_img = cv.imread('/DATA_17/hjjo/YOLOX_pruning/YOLOX/test_image.jpeg', cv.IMREAD_COLOR)
#     #바운딩 박스 그리기
#     ori_img = drawBoundingBox(ori_img,original_bboxes,original_cls)
    
#     dir_path = './output'
#     file_path = os.path.join(dir_path, '1.jpeg')
#     cv2.imwrite(file_path,ori_img)
    print('complete')


