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
import common
import vis
import coco_classes


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

def preproc(img, input_size, swap=(2, 0, 1)):
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
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img



def img_process(img_path,batch_size):
    ori_img = cv.imread(img_path, cv.IMREAD_COLOR)
    img = preproc(ori_img,(640,640), swap=(2, 0, 1))
    img_list = [img for _ in range(batch_size)]
    img_stack = np.stack(img_list, axis=0)
    return img_stack, ori_img



def make_output(result,batch_size):
    num_classes = 80
    confthre =0.5
    nmsthre  =0.3
    result = np.reshape(result,(batch_size,1,-1))    
    outputs = torch.Tensor(result)
    outputs = outputs.view([batch_size, -1,num_classes+5]) 
    outputs = postprocess(
        outputs, num_classes, confthre,
        nmsthre, class_agnostic=True)

    return outputs


def visual(img, output, ratio, cls_conf=0.35):

    if output is None:
        return img
    output = output.cpu()

    bboxes = output[:, 0:4]

    # preprocessing: resize
    bboxes /= ratio

    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]

    vis_res = vis.vis(img, bboxes, scores, cls, cls_conf, coco_classes.COCO_CLASSES)

    return vis_res




#엔진 경로 설정해주기 
trt_engine_path = '/DATA_17/ij/trt_inference/yolox_m.trt'

#배치사이즈 설정하기
batch_size = 1
#이미지 경로 설정     
img_path = '/DATA_17/ij/test_image.jpeg'   

save_img_path = '/DATA_17/ij/test.jpeg'   

#Engine class 소환 
Engine = common.Engine()      
#inference에 필요한 context 만들어주기  
engine ,context, stream = Engine.make_context(trt_engine_path, batch_size)

#버퍼 할당해주기 
inputs = Engine.allocate_buffers(engine, batch_size, 'input')
output = Engine.allocate_buffers(engine, batch_size, 'output')

img_stack, ori_img = img_process(img_path,batch_size)


average = 0
loop_cnt = 100
for i in range(loop_cnt):
    s3= time.time()
    input_data = torch.tensor(img_stack).cuda() #input 버퍼할당해 주지 않고 데이터를 바로 보낼때 
    result = Engine.do_inference_v2(context, input_data, None, output, stream) #결과 생성
    e3 = time.time()
    if i < 2:
        continue
    print('time',e3 - s3)
    average += e3 - s3
print('평균 시간 : ',average/(loop_cnt-2))


#postprocess
outputs = make_output(result,batch_size)
print(outputs)

ratio = min(640 / ori_img.shape[0], 640 / ori_img.shape[1])
result_image = visual(ori_img, outputs[0], ratio, cls_conf=0.35)
cv.imwrite(save_img_path, result_image)


# outputs 예시 (박스,score,score,class)
# [tensor([[390.1489,  48.4131, 547.3511, 422.5869,   0.9863,   0.9541,   0.0000],
#         [ 51.7764,  47.5381, 259.2236, 421.7119,   0.9883,   0.9478,   0.0000],
#         [221.2948,  35.4743, 406.7052, 421.5256,   0.9897,   0.9360,   0.0000],
#         [293.8126, 137.6245, 319.6874, 198.3755,   0.9800,   0.8730,  27.0000],
#         [467.4181, 137.4220, 494.5819, 223.3280,   0.9419,   0.8882,  27.0000]])




