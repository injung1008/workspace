## detect.py

import argparse
import os
import sys
import numpy as np
import tensorrt as trt
import cv2
import torchvision
import torch
import time
import traceback
import importlib
import common

# common = importlib.import_module('.'+str('common'), package='interfaces')


class Detector:

    def __init__(self, logger):
#     def __init__(self,weights):       
        self.logger = logger
#         self.batch_size = 64
        
        self.num_classes = 80
        self.confthre = 0.5
        self.nmsthre = 0.3
        
        
        self.input_h = 640
        self.input_w = 640
        
        
    def init_sample_data(self):
        try:
            base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            sample_path = os.path.join(base_path, 'sample.png')
            sample_img = cv2.imread(sample_path)

            sample_img_batch = [sample_img] * self.batch_size
            sample_img_batch = torch.tensor(sample_img_batch).cuda()
            
            self.inference_batch(sample_img_batch)
        except Exception as e:
            self.logger.error(f'initialize sample image : {e}')
            self.logger.error(traceback.format_exc())
        
        
    def postprocess(self,prediction, class_agnostic=False):

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
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + self.num_classes], 1, keepdim=True)

            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= self.confthre).squeeze()
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            if not detections.size(0):
                continue
            
            if class_agnostic:
                nms_out_index = torchvision.ops.nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    self.nmsthre,
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

    
    
    def preprocess(self,frame_batch) :  
        result = torch.zeros([len(frame_batch), 3, self.input_h, self.input_w], dtype=torch.float32, device=torch.device("cuda:0")).fill_(144)
        for idx, frame in enumerate(frame_batch) :
#             frame = frame.reshape(frame.size()[1], frame.size()[2], frame.size()[0]).permute(2,0,1).to(torch.device("cuda:0"))
            frame = torch.from_numpy(frame).to(torch.device("cuda:0")).permute(2,0,1)
#             frame = frame.reshape(frame.size()[1], frame.size()[2], frame.size()[0])
#             frame = frame.permute(2,0,1)
            _, h, w = frame.shape
            r = min(self.input_h/h, self.input_w/w)
            rw, rh = int(r*w), int(r*h)
            pad_w, pad_h = (self.input_w-rw), (self.input_h-rh)
            resized_img = torchvision.transforms.functional.resize(frame, (rh,rw)).float()
            result[idx, :,:rh,:rw] = resized_img      
#             torch.cuda.empty_cache()
        return result




#     def preprocess(self,frame_batch) :
#         t1 = time.time()
#         result = torch.zeros([len(frame_batch), 3, self.input_h, self.input_w], dtype=torch.float32, device=torch.device("cuda:0")).fill_(144)
#         t2 = time.time()
#         self.logger.info(f'torch.zeros time : {t2-t1}, ptr : {result.data_ptr()}')


#         for idx, frame in enumerate(frame_batch) :
#             t1 = time.time()
#             frame = torch.from_numpy(frame).to(torch.device("cuda:0")).permute(2,0,1)
#             t2 = time.time()
#             self.logger.info(f'torch.from_numpy time : {t2-t1}')

#             _, h, w = frame.shape

#             r = min(self.input_h/h, self.input_w/w)
#             rw, rh = int(r*w), int(r*h)
#             pad_w, pad_h = (self.input_w-rw), (self.input_h-rh)
#             t1 = time.time()
#             resized_img = torchvision.transforms.functional.resize(frame, (rh,rw)).float()
#             t2 = time.time()
#             self.logger.info(f'torchvision.transforms.functional.resize time : {t2-t1}')
#             t1 = time.time()
#             result[idx, :,:rh,:rw] = resized_img
#             t2 = time.time()
#             self.logger.info(f'result[idx, :,:rh,:rw] time : {t2-t1}')

#         return result




    
    def load(self, trt_engine_path='/DATA_17/hjjo/selftest/Model_Management/engines/yoloxm_fp16_032.trt'):
        self.weights = trt_engine_path
        self.Engine = common.Engine()      
        self.Engine.make_context(self.weights)
        self.batchsize = int(self.Engine.input_shape[0])
#         self.init_sample_data()


    def inference(self,input_data) : 
        output_data = self.Engine.do_inference_v2(input_data)
        return output_data

    
    def inference_batch(self, org_frame_batch):
        
        t0 = time.time()
        frame_data = self.preprocess(org_frame_batch)
        t1 =  time.time()
        results_ori = self.inference(frame_data)

        t2 = time.time()
        results = self.postprocess(results_ori,class_agnostic=True)
        t3 = time.time()
        
        self.logger.info(f'yolox 1.preprocess : {t1 - t0}, 2.inference : {t2 - t1}, 3.postprocess : {t3 - t2},  4.total : {t3 - t0}')
#         if len(results) > 0 :
        self.logger.info(f'yolox results : {results[0]}')
        return results
def module_load(logger):
    dt = Detector(logger)
    return dt
# if __name__ == '__main__':
#     torch.cuda.init() #파이프 라인 필수 
#     trt_engine_path = '/xaiva/model/yolox_trt/weights/yoloxm_int8_032.trt'
#     img_path = '/xaiva/model/yolox_trt/test_image.jpeg'
#     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#     img = torch.from_numpy(img).to(torch.device("cuda"))
#     print(img.shape)
#     img_list = [img for _ in range(32)]

#     yolox =Detector(trt_engine_path)
    
#     #engine load
#     yolox.load()

    
#     #img stack
#     s = time.time()
#     input_data = yolox.preprocess(img_list)
#     print(f' prepoc : {time.time() - s}')

#     #inference
#     s1 = time.time()
#     output_data = yolox.inference(input_data)
#     print(f' inference : {time.time() - s1}')

#     #postprocess
#     s2 = time.time()
#     outputs = yolox.postprocess(output_data, class_agnostic=True)
#     print(f' postprocess : {time.time() - s2}')
    
#     print(f'total_time : {time.time() - s}')
#     print(outputs)        

COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)   
    

COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)