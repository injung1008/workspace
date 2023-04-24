import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import time
ctx = cuda.Device(0).make_context()

class Engine:

    def load_engine(self, runtime,engine_path):
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
        return engine
    
    
    def make_context(self,trt_engine_path, batch_size,input_w,input_h):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        #엔진 로드 
        engine = self.load_engine(runtime,trt_engine_path)
        #inference를 위한 context 만들기
        context = engine.create_execution_context()
        #context 사이즈 지정 해주기
        context.set_binding_shape(0, (batch_size, 3,input_w,input_h)) #바인딩의 dynamic shape을 설정한다 
        
        stream = cuda.Stream()

        return engine ,context, stream
    

        # input / output buffer 생성
    def allocate_buffers(self, engine, batch_size, buffer_type): 
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
    
    
    def do_inference_v2(self,context, input_data, inputs, output, stream):
        s = time.time()
        ctx.push()
        bindings = None     
        if inputs == None : # 입력 버퍼 할당 하지 않은경우(input_data == tensor cuda)
            bindings = [
                int(input_data.contiguous().data_ptr())
                ,int(output['device_mem'])
            ]        
        else :
            bindings = [
                int(input['device_mem'])
                ,int(output['device_mem'])
            ]        

            inputs['host_mem'] = input_data        
            cuda.memcpy_htod_async(inputs['device_mem'], inputs['host_mem'], stream)

        context.execute_async_v2(bindings,stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output['host_mem'], output['device_mem'], stream)

        stream.synchronize()
        ctx.pop()
        e = time.time()
#         print('infe',e-s)
        return output['host_mem']

      
