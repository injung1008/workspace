import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import numpy as np
import time

class Engine:
    def __init__(self):
        self.buffer = dict() 
        self.ctx = cuda.Device(0).make_context()
    
    def load_engine(self, runtime,engine_path):
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
        return engine
 
    
    def make_context(self,trt_engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        #엔진 로드 
        self.engine = self.load_engine(runtime,trt_engine_path)
        #inference를 위한 context 만들기
        self.context = self.engine.create_execution_context()
        #context 사이즈 지정 해주기
        self.stream = cuda.Stream()



    def allocate_buffers(self,buffer_type): 

        for binding in self.engine:
            b_shape = self.engine.get_binding_shape(binding)
            b_shape[0] = self.max_batch
            size = trt.volume(b_shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding)) #numpy.float32

            if buffer_type == 'input':
                if self.engine.binding_is_input(binding):
                    host_mem = cuda.pagelocked_empty(size, dtype)
                    device_mem = cuda.mem_alloc(host_mem.nbytes)
                else:
                    continue            

            if buffer_type == 'output':
                if self.engine.binding_is_input(binding):
                    continue
                else:
                    host_mem = cuda.pagelocked_empty(size, dtype)
                    device_mem = cuda.mem_alloc(host_mem.nbytes)

        buffer = {
            'host_mem' : host_mem
            ,'device_mem' : device_mem
        }
        return buffer
    
    def allocate_buffers_all(self):
        self.max_batch = self.engine.get_profile_shape(0,'input')[2][0]
        self.input_w = self.engine.get_profile_shape(0,'input')[2][3]
        self.input_h = self.engine.get_profile_shape(0,'input')[2][2]
        self.buffer[self.max_batch] = dict()
        self.buffer[self.max_batch]['input'] = self.allocate_buffers('input')
        self.buffer[self.max_batch]['output'] = self.allocate_buffers('output')
#         shape torch.Size([1, 3, 256(h), 128(w)])
        return self.max_batch, self.input_w, self.input_h
        
    
    
    def do_inference_v2(self, input_data):
        img_batch = input_data.shape[0]       
        self.context.set_binding_shape(0, (img_batch, 3, self.input_h,self.input_w))             
        if isinstance(input_data, np.ndarray) :
            use_cpu = True
        else :
            use_cpu = False

    
        self.ctx.push()
        bindings = None     
        if use_cpu == False : 
            bindings = [
                int(input_data.contiguous().data_ptr())
                ,int(self.buffer[self.max_batch]['output']['device_mem'])
            ] 
        
        else :
            bindings = [
                int(self.buffer[self.max_batch]['input']['device_mem'])
                ,int(self.buffer[self.max_batch]['output']['device_mem'])
            ]        

            self.buffer[self.max_batch]['input']['host_mem'] = input_data        
            cuda.memcpy_htod_async(self.buffer[self.max_batch]['input']['device_mem'], self.buffer[self.max_batch]['input']['host_mem'], self.stream)
        
        self.context.execute_async_v2(bindings,stream_handle=self.stream.handle)
#         cuda.memcpy_dtoh_async(self.buffer[self.max_batch]['output']['host_mem'], self.buffer[self.max_batch]['output']['device_mem'], self.stream)

        self.stream.synchronize()
        self.ctx.pop()

#         return self.buffer[self.max_batch]['output']['host_mem'],img_batch
        return None
        
    def flush(self):
        while self.ctx.get_current() is not None:
            self.ctx.pop()
            time.sleep(0.005)
        self.ctx.detach()
        del(self.ctx)


