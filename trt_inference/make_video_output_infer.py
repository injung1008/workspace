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

from loguru import logger

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
#     padded_img = np.ascontiguousarray(padded_img, dtype=np.float32) / 255.0
    return padded_img



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



def visual(img,output, ratio, cls_conf=0.35):

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


def img_process(img,batch_size):
    img = preproc(img,(640,640), swap=(2, 0, 1))
    img_list = [img for _ in range(batch_size)]
    img_stack = np.stack(img_list, axis=0)
    return img_stack







#비디오 읽고, 전체적인 코드 실행 
def proc(video_path, output_file,context, output, stream):
    cap = cv.VideoCapture(video_path)
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv.CAP_PROP_FPS)
    
    save_path = output_file
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv.VideoWriter(
        save_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    print('fps',fps)

    cnt = 0

    if cap.isOpened() :    
        while True:
            cnt+=1
            #원본이미지 = frame 
            ret, frame = cap.read()
            if ret == False :
                break
            if cnt < 0:
                continue
#             프레임수 특정값 이상일떄는 break 할때 
#             if cnt > 1:
#                 break
            #현재 프레임
            num_frame = round(cap.get(cv.CAP_PROP_POS_FRAMES)) - 1
            print('frame : ',num_frame)
            

            img_stack = img_process(frame,batch_size)

            #input 버퍼할당해 주지 않고 데이터를 바로 보낼때 
            input_data = torch.tensor(img_stack).cuda() 

            #결과 생성
            result = Engine.do_inference_v2(context, input_data, None, output, stream) 

            #postprocess
            outputs = make_output(result,batch_size)

            ratio = min(640 / frame.shape[0], 640 / frame.shape[1])

            result_image = visual(frame,outputs[0], ratio, cls_conf=0.35)
            
            vid_writer.write(result_image)
            ch = cv.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
           


            
#엔진 경로 설정해주기 
# trt_engine_path = '/DATA_17/ij/trt_inference/convert_modules/yolov5l.trt'
trt_engine_path = '/DATA_17/ij/trt_inference/yolox_x.trt'
#배치사이즈 설정하기
batch_size = 1

#Engine class 소환 
Engine = common.Engine()      
#inference에 필요한 context 만들어주기  
engine ,context, stream = Engine.make_context(trt_engine_path, batch_size)

#버퍼 할당해주기 
inputs = Engine.allocate_buffers(engine, batch_size, 'input')
output = Engine.allocate_buffers(engine, batch_size, 'output')

intput_path = './input/coco.mp4'
output_path = './output/coco_YOLOX_X_trt..mp4'
proc(intput_path,output_path,context, output, stream)
print('#####################done########################')
