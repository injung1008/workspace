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

        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

                
                
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
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * 1
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
       
            
    def __call__(self,x:np.ndarray,batch_size=2):
        print(x.shape)
        x = x.astype(self.dtype)
        np.copyto(self.inputs[0].host,x.ravel())
        
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        
        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
            #print('self.outputs.shape',out.host.reshape(batch_size,-1))
            
        
        self.stream.synchronize()

        cnt = 0
        for out in self.outputs :
            cnt += 1
            out.host.reshape(batch_size,-1) 
#         print('cnt',cnt)
        return [out.host.reshape(batch_size,-1) for out in self.outputs]

# trt_engine_path = '/DATA_17/ij/yolox_s_model.trt'
# model = TrtModel(trt_engine_path)
    

if __name__ == "__main__":
 
    batch_size = 1
    trt_engine_path = 'resnet50_2_4.trt'
    model = TrtModel(trt_engine_path)
    shape = model.engine.get_binding_shape(0)
    inputs, outputs, bindings, stream = model.allocate_buffers()

#     data = np.random.randint(0,255,(batch_size,*shape[1:]))/255
#     print(data1.shape) # (426, 640, 3)
#     print(data.shape) #(1, 3, 640, 640)

    preproc = ValTransform(legacy=False)   
    img = cv.imread('/DATA_17/hjjo/YOLOX_pruning/YOLOX/test_image.jpeg', cv.IMREAD_COLOR)
    img, _ = preproc(img, None, (224,224))
    for _ in range(10) :
        img2 =  np.concatenate((img, img), axis=0)
    print(img2.shape)
#     t0 = 0
    for i in range(50):
        s = time.time()
        result = model(img,batch_size)
        #print('result',result)
        e = time.time()
#         end = e-s
#         t0 += end
        print(f'inference time : {e-s}')
#         print(t0)
#     print('평균시간',t0/10)

# result [array([[1.7333633e+01, 5.3959584e+00, 2.8746363e+01, ..., 1.9049942e-03,
#         1.7058253e-03, 1.9973218e-03]], dtype=float32)]

