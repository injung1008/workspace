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
import os
import csv
import argparse

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trt_dir", default='fp16')
    parser.add_argument("--input_w", type=int,default=640)
    parser.add_argument("--input_h", type=int,default=640)
    parser.add_argument("--img_path", default='/DATA_17/trt_test/test/test_image.jpeg')
    return parser

args = make_parser().parse_args()


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



def test(trt_engine_path,img_path,batch_size,loop_cnt,input_w,input_h):

    #Engine class 소환 
    Engine = common.Engine()      
    #inference에 필요한 context 만들어주기  
    engine ,context, stream = Engine.make_context(trt_engine_path, batch_size,input_w,input_h)

    #버퍼 할당해주기 
    inputs = Engine.allocate_buffers(engine, batch_size, 'input')
    output = Engine.allocate_buffers(engine, batch_size, 'output')
    img_stack, ori_img = img_process(img_path,batch_size)

    for i in range(3):
        input_data = torch.tensor(img_stack).cuda() #input 버퍼할당해 주지 않고 데이터를 바로 보낼때 
        result = Engine.do_inference_v2(context, input_data, None, output, stream) #결과 생성
  
    total_time = 0
    s3 = time.time()
    for i in range(loop_cnt):
        input_data = torch.tensor(img_stack).cuda() #input 버퍼할당해 주지 않고 데이터를 바로 보낼때 
        result = Engine.do_inference_v2(context, input_data, None, output, stream) #결과 생성
    e3 = time.time()
    total_time += e3 - s3
    
#     print('평균 시간 : ',total_time/(loop_cnt))
    average_total_time = total_time/(loop_cnt)

    #postprocess
    outputs = make_output(result,batch_size)

    out = outputs[0].shape[0]
    
    #save_img_path = '/DATA_17/ij/test.jpeg'  
    # ratio = min(640 / ori_img.shape[0], 640 / ori_img.shape[1])
    # result_image = visual(ori_img, outputs[0], ratio, cls_conf=0.35)
    # cv.imwrite(save_img_path, result_image)

    return round(average_total_time,5), batch_size, loop_cnt, out

###############################실행##########################################



#이미지 경로 설정     
img_path = args.img_path


import os
trt_dir = args.trt_dir
targetDir = f'../engines/{trt_dir}/'

#이미지 경로 설정     
img_path = args.img_path

csv_save_path = f'./csv/{trt_dir}.csv'

input_w = args.input_w
input_h = args.input_h

##targetDir에서 .xml파일 이름들 리스트로 가져오기
trt_file_list = os.listdir(targetDir)

trt_list = []
for file in trt_file_list:
    if '.trt' in file:
        trt_list.append(file)

trt_list = sorted(trt_list)


with open(csv_save_path, 'w', encoding='UTF-8') as f:
    w = csv.writer(f)
    title = "Model file_name trt_type make_batch test_batch average_time time_per_img output=5 loop_cnt".split(" ")
    w.writerow(title)


    for trt_file in trt_list:
        engine_target_path = targetDir + trt_file
        

        f_name = trt_file.split('_')
        max_batch_size = int(f_name[2][:-4])
        trt_type = f_name[1]
        model_name = f_name[0]

        test_batch = [1]
        
        middle = int(max_batch_size/2)

        if middle > 10 :
            test_batch.append(int(middle/2))
            test_batch.append(int(middle))
            test_batch.append(int(middle/2 + middle))
            test_batch.append(max_batch_size)
        else : 
            test_batch.append(middle)
            test_batch.append(max_batch_size)
        best_time = []

#5개만 돌릴때       
#         for batch_size in test_batch :

#1개씩 돌릴때 
        for batch_size in range(1,max_batch_size+1) :
            loop_cnt = int(2000/batch_size) + 3 
            average_total_time,batch_size,loop_cnt,out = test(engine_target_path,img_path,batch_size,loop_cnt,input_w,input_h)
            print('trt_file :',trt_file,'batch_size: ',batch_size)
            print('average_time: ',average_total_time/batch_size)
            
            time_per_img = round(average_total_time/batch_size,5)
            
            best_time.append(time_per_img)
            
            data = f'{model_name} {trt_file} {trt_type} {max_batch_size} {batch_size} {average_total_time} {time_per_img} {out} {loop_cnt}'.split(" ")
            w.writerow(data)
        
        t_min = min(best_time)
#         tmin_index = best_time.index(t_min) 
#         best_batch = test_batch[tmin_index]
        tmin_index = best_time.index(t_min) +1
        best_batch = tmin_index
        s = '_'
        t = 'best_batch'
        p = 'time_per_img(batch)'
        data = f'{s} {t} {best_batch} {p} {t_min} {s} {s} {s} {s}'.split(" ")
        w.writerow(data)



    
    
    

    
    
    