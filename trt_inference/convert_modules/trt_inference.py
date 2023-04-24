import argparse
import os
import sys
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import cv2 as cv
import torchvision
import torch
import time

# from utils.general import non_max_suppression

ctx = cuda.Device(0).make_context()


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        if not image_pred.size(0):
            continue
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output

def preproc(img,input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    
    resized_img = cv.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    padded_img = padded_img.transpose(swap)
#     padded_img = np.ascontiguousarray(padded_img, dtype=np.float32) / 255.0
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img



def img_process(img_path,batch_size):
    ori_img = cv.imread(img_path, cv.IMREAD_COLOR)
    img = preproc(ori_img,(640,640), swap=(2, 0, 1))
    img_list = [img for _ in range(batch_size)]
    img_stack = np.stack(img_list, axis=0)
    return img_stack

# input / output buffer 생성
def allocate_buffers(engine, batch_size, buffer_type="output"): 
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
      

def do_inference_v2(context, input_data, input, output, stream):
    res = None
    ctx.push()
    bindings = None     
    if input == None : # 입력 버퍼 할당 하지 않은경우(input_data == tensor cuda)
        bindings = [
            int(input_data.contiguous().data_ptr())
            ,int(output['device_mem'])
        ]        
    else :
        bindings = [
            int(input['device_mem'])
            ,int(output['device_mem'])
        ]        
        
        input['host_mem'] = input_data        
        cuda.memcpy_htod_async(input['device_mem'], input['host_mem'], stream)
    
    context.execute_async_v2(bindings,stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(output['host_mem'], output['device_mem'], stream)

    stream.synchronize()
    ctx.pop()

    res = output['host_mem']
    return res

def make_context(trt_engine_path, batch_size):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    #엔진 로드 
    engine = load_engine(runtime, trt_engine_path)
    #inference를 위한 context 만들기
    context = engine.create_execution_context()
    #context 사이즈 지정 해주기
    context.set_binding_shape(0, (batch_size, 3, 640, 640)) #바인딩의 dynamic shape을 설정한다 
    stream = cuda.Stream()
    
    return engine ,context, stream



def load_engine(trt_runtime, engine_path):
    trt.init_libnvinfer_plugins(None, "")             
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine

def make_output(result,batch_size):
    result = np.reshape(result,(batch_size,1,-1))    
    outputs = torch.Tensor(result)
    outputs = outputs.view([batch_size, -1,85]) 
    num_classes = 80
    confthre =0.5
    nmsthre  =0.3
    outputs = postprocess(
        outputs, num_classes, confthre,
        nmsthre, class_agnostic=True)

    return outputs


start = time.time()
#엔진 경로 설정해주기 
trt_engine_path = '/DATA_17/ij/trt_inference/yolox_l.trt'
# trt_engine_path = '/data/ij/trt_infer/convert_modules/yolov5s_fp16_module.trt'

#이미지 경로 설정     
img_path = '/DATA_17/ij/test_image.jpeg'

#배치사이즈 설정하기
batch_size = 1


engine ,context, stream = make_context(trt_engine_path, batch_size)

#버퍼 할당해주기 
inputs = allocate_buffers(engine, batch_size, buffer_type="input")
output = allocate_buffers(engine, batch_size, buffer_type="output")

img_stack = img_process(img_path,batch_size)


average = 0
loop_cnt = 100
for i in range(loop_cnt):
    s3= time.time()
    input_data = torch.tensor(img_stack).cuda() #input 버퍼할당해 주지 않고 데이터를 바로 보낼때 
    result = do_inference_v2(context, input_data, None, output, stream) #결과 생성
    e3 = time.time()
    if i < 2:
        continue
    print('time',e3 - s3)
    average += e3 - s3
print('평균 시간 : ',average/(loop_cnt-2))

output = make_output(result,batch_size)

print(output)



print("done")

end = time.time()


print('전체 시간 : ',end - start) 
ctx.pop()