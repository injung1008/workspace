import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import numpy as np
import time

class Engine:
    def __init__(self):
        self.res = dict()
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
        context = self.engine.create_execution_context()
        #context 사이즈 지정 해주기
        stream = cuda.Stream()

        return self.engine, context, stream, self.ctx
    

#     def allocate_buffers_all(self,max_batch,engine):
#         for i in range(1,max_batch+1):
#             self.res[i] = dict()
#             self.res[i]['input'] = allocate_buffers(engine, i, 'input')
#             self.res[i]['output'] = allocate_buffers(engine, i, 'output')
    
    
    def allocate_buffers_all(self,max_batch,engine):
        for i in range(64,65):
            self.res[i] = dict()
            self.res[i]['input'] = allocate_buffers(engine, i, 'input')
            self.res[i]['output'] = allocate_buffers(engine, i, 'output')
        print(self.res)
            
            
    
    def infer(self,context, input_data, use_cpu, stream,input_w,input_h,batch):
        context.set_binding_shape(0, (batch, 3, input_w,input_h))
#         self.ctx.push()
        bindings = None     
#         if use_cpu == False : 
#             bindings = [
#                 int(input_data.contiguous().data_ptr())
#                 ,int(self.res[batch]['output']['device_mem'])
#             ] 
        if use_cpu == False : 
            bindings = [
                int(input_data.contiguous().data_ptr())
                ,int(self.res[batch*2]['output']['device_mem'])
            ] 
        else :
            bindings = [
                int(self.res[batch]['input']['device_mem'])
                ,int(self.res[batch]['output']['device_mem'])
            ]        

            self.res[batch]['input']['host_mem'] = input_data        
            cuda.memcpy_htod_async(self.res[batch]['input']['device_mem'], self.res[batch]['input']['host_mem'], stream)
        
        context.execute_async_v2(bindings,stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(self.res[batch*2]['output']['host_mem'], self.res[batch*2]['output']['device_mem'], stream)

        stream.synchronize()
#         self.ctx.pop()
        return self.res[batch*2]['output']['host_mem']
    
    
    
    def do_inference_v2(self,context, input_data, stream,input_w,input_h):
        #들어온 input이 np일 경우 cpu 사용 
        self.ctx.push()

        if isinstance(input_data, np.ndarray) :
            use_cpu = True
        else :
            use_cpu = False
          
        total_batch = input_data.shape[0] #input img 배치 사이즈
        mini_batch = self.engine.get_profile_shape(0,'input')[1][0]  #trt생성 최적 배치 사이즈
        total_output = []

        portion = int(total_batch/mini_batch) #몫 - for 돌아야함
        remain = int(total_batch%mini_batch) #나머지 - 나머지 값 만큼 더 돌기
        
        s = time.time()

        results = self.infer(context, input_data, use_cpu, stream,input_w,input_h,32)
        e = time.time()
        print('infer_time',e-s)

#         s = time.time()
#         for i in range(portion):
#             self.infer(context, input_data[i*mini_batch], use_cpu, stream,input_w,input_h,mini_batch)
#         if remain > 0 :
#             self.infer(context, input_data[total_batch-remain], use_cpu, stream,input_w,input_h,remain)
#         e = time.time()
#         print('infer_time',e-s)
        
#         results = None
#         s1 = time.time()
#         results = np.concatenate(total_output)
#         e1 = time.time()
#         print('cocat_time',e1-s1)
        
        self.ctx.pop()
        return results
    
#     def do_inference_v2(self,context, input_data, use_cpu, stream,input_w,input_h):
#         img_batch = input_data.shape[0]
#         context.set_binding_shape(0, (img_batch, 3, input_w,input_h))
#         self.ctx.push()
#         bindings = None     
#         if use_cpu == None : 
#             bindings = [
#                 int(input_data.contiguous().data_ptr())
#                 ,int(self.res[img_batch]['output']['device_mem'])
#             ] 
        
#         else :
#             bindings = [
#                 int(self.res[img_batch]['input']['device_mem'])
#                 ,int(self.res[img_batch]['output']['device_mem'])
#             ]        

#             self.res[img_batch]['input']['host_mem'] = input_data        
#             cuda.memcpy_htod_async(self.res[img_batch]['input']['device_mem'], self.res[img_batch]['input']['host_mem'], stream)
        
#         context.execute_async_v2(bindings,stream_handle=stream.handle)
#         cuda.memcpy_dtoh_async(self.res[img_batch]['output']['host_mem'], self.res[img_batch]['output']['device_mem'], stream)

#         stream.synchronize()
#         self.ctx.pop()

#         return self.res[img_batch]['output']['host_mem']

        

      
    # input / output buffer 생성
def allocate_buffers(engine, batch_size, buffer_type): 
    res = None
    host_mem = None
    device_mem = None

    for binding in engine:
        b_shape = engine.get_binding_shape(binding)
        b_shape[0] = batch_size
        size = trt.volume(b_shape)
        dtype = trt.nptype(engine.get_binding_dtype(binding)) #numpy.float32

        if buffer_type == 'input':
            if engine.binding_is_input(binding):
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
            else:
                continue            

        if buffer_type == 'output':
            if engine.binding_is_input(binding):
                continue
            else:
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)

    res = {
        'host_mem' : host_mem
        ,'device_mem' : device_mem
    }
    return res
