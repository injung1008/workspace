import tensorrt as trt
import numpy as np
import time
import torch
import os 

class Engine:
    def __init__(self):
        pass

    def load_engine(self, runtime,engine_path):
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
        return engine
 
    
    def make_context(self,trt_engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)

        self.engine = self.load_engine(runtime,trt_engine_path)

        self.context = self.engine.create_execution_context()

        self.input_shape = self.engine.get_profile_shape(0,0)[2]
        self.out_shape = self.engine.get_binding_shape(1)

    
    def do_inference_v2(self, input_data):
        img_batch = input_data.shape[0]  

        self.out_shape[0] = img_batch
        self.input_shape[0] = img_batch

        output_data = torch.empty(size=tuple(self.out_shape), dtype=torch.float32, device=torch.device("cuda:0"))

        self.context.set_binding_shape(0, tuple(self.input_shape))             

        bindings = None     

        bindings = [
            int(input_data.contiguous().data_ptr())
            ,int(output_data.data_ptr()) 
        ] 

        self.context.execute_async_v2(bindings,stream_handle=torch.cuda.current_stream().cuda_stream)   

        return output_data
      
        



