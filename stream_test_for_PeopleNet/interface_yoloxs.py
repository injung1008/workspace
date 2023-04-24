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
import torch.cuda as cuda

class DETECTOR:

    def __init__(self):      
#         self.logger = logger
        
        self.num_classes = 80
        self.confthre = 0.5
        self.nmsthre = 0.3

        
        self.input_h = 640
        self.input_w = 640
        self.device = torch.device('cuda:0')
        
        
        self.categories = {} 
        # Sports
        self.categories['1'] = [35,36,30,37,31,32,33,38,39] 
        # Vehicle
        self.categories['2'] = [5,2,9,6,3,4,7,8]
        # Food
        self.categories['3'] = [48,47,51,56,52,55,53,50,54,49]
        # Product
        self.categories['4'] = [60,74,40,46,68,57,75,42,61,43,79,67,34,44,64,69,70,59,73,66,77,72,58,45,78,71,80,63,76,41]
        # Animal
        self.categories['5'] = [22,15,16,20,17,21,24,18,65,1,19,23]
        # Structure
        self.categories['6'] = [14,11,13,12,62,10]
        # Accessory
        self.categories['7'] = [25,27,29,28,26]
        # person
        self.categories['8'] = [1]
        
        
        
        
        self.category_id = '8'
#         self.stream = cuda.stream 
        

#         self.s1 = cuda.Stream(device="cuda")
#         self.s2 = cuda.Stream(device="cuda")
#         self.s3 = cuda.Stream(device="cuda")
    
    def load_set_meta(self, channel_id=None, model_parameter=None, channel_info_dict=None, model_name=None):
        pass
#         category_id = model_parameter['category_id']
#         channel_info_dict[channel_id]['map_data'][model_name]['category_id'] = category_id
        
    
    def load(self, weights):
        self.weights = weights
        self.Engine = common.Engine()      
        self.Engine.make_context(self.weights)
        self.batchsize = int(self.Engine.input_shape[0])
        print('model load done')
        
#         self.init_sample_data()
        
        
    def init_sample_data(self):
        try:
            base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            sample_path = os.path.join(base_path, 'sample.png')
            sample_img = cv2.imread(sample_path)

            sample_img_batch = [sample_img] * self.batch_size
            sample_img_batch = torch.tensor(sample_img_batch).cuda()
            
            self.inference_batch(sample_img_batch)
        except Exception as e:
#             self.logger.error(f'initialize sample image : {e}')
#             self.logger.error(traceback.format_exc())
            pass

    def parse_input(self,input_data_batch):
        res = []
        for input_data in input_data_batch:
            frame = input_data['framedata']['frame']
            bbox = input_data['bbox']
            cropped_img = common.getCropByFrame(frame,bbox)
            res.append(cropped_img)
        return res
       

    
    
    def preprocess(self,frame_batch) :  
#         result = torch.zeros([len(frame_batch), 3, self.input_h, self.input_w], dtype=torch.float32, device=torch.device("cuda:0")).fill_(144)
        result = torch.empty([len(frame_batch), 3, self.input_h, self.input_w], dtype=torch.float32, device=torch.device("cuda:0"))
        scale_list = []
        for idx, frame in enumerate(frame_batch) :

#             frame = frame.to(torch.device("cuda:0"))
            _, h, w = frame.shape
            
            r = min(self.input_h/h, self.input_w/w)
            if r < 1 :  
                rw, rh = int(r*w), int(r*h)

                resized_img = torchvision.transforms.functional.resize(frame, (rh,rw)).float()
                result[idx, :,:rh,:rw] = resized_img 
                scale_list.append(r)
            else : 

                result[idx, :,:h,:w] = frame
                scale_list.append(None)

            
        return result, scale_list

    
    def preprocess_for_calibrator(self,frame_batch) :  
        result = torch.zeros([len(frame_batch), 3, self.input_h, self.input_w], dtype=torch.float32, device=torch.device("cpu")).fill_(144)
        scale_list = []
        for idx, frame in enumerate(frame_batch) :

#             frame = frame.to(torch.device("cuda:0"))
            _, h, w = frame.shape
            
            r = min(self.input_h/h, self.input_w/w)
            if r < 1 :  
                rw, rh = int(r*w), int(r*h)

                resized_img = torchvision.transforms.functional.resize(frame, (rh,rw)).float()
                result[idx, :,:rh,:rw] = resized_img 
                scale_list.append(r)
            else : 

                result[idx, :,:h,:w] = frame
                scale_list.append(None)

            
        return result, scale_list



    def inference(self,input_data) : 
        output_data = self.Engine.do_inference_v2(input_data)
        return output_data[0]


    def postprocess(self,prediction,scale_list, class_agnostic=False):

        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]



        outputs = list()
        for i, image_pred in enumerate(prediction):
            if not image_pred.size(0):
                outputs.append(None)
                continue
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + self.num_classes], 1, keepdim=True)

            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= self.confthre).squeeze()
            detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            if not detections.size(0):
                outputs.append(None)
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
                    self.nmsthre,
                )

            detections = detections[nms_out_index]
            

            s = time.time()
            ### bbox 복원
            output = list()
            for j, det in enumerate(detections):
                if isinstance(det, type(None)):
                    output.append(None)
                    continue

                tmp = det.detach().cpu().numpy()
                label = str(int(tmp[6]))
                    
                x1, y1, x2, y2 = map(int, tmp[:4])
                t2 = time.time()

                restore_bbox = common.restoreBboxScale( [x1,y1,x2,y2], (scale_list[i],scale_list[i]) )
                t3 = time.time()

                tmp[0] = restore_bbox[0]
                tmp[1] = restore_bbox[1]
                tmp[2] = restore_bbox[2]
                tmp[3] = restore_bbox[3]
                
                                
                score1 = tmp[4]
                score2 = tmp[5]
                score = (tmp[4] + tmp[5]) / 2
                
                x1, y1, x2, y2 = map(int, restore_bbox)            
                bbox = [x1, y1, x2, y2]
                output.append({"bbox":bbox, "score":score, "label":label})
                t4 = time.time()
            outputs.append(output)
            e = time.time()
           
        return outputs             
    
    def parse_output(self,input_data_batch,output_batch,reference_CM):
        res = []
        
        
        for idx_i, data in enumerate(input_data_batch): 
            framedata = data['framedata']
            scenario = data['scenario']
            channel_id = framedata['meta']['source']['channel_id']
#             category_id = reference_CM.channel_info_dict[channel_id]['map_data']['yolox_detect_yoloxx']['category_id']
            category_id = self.category_id
            
            category_id_list = category_id.split(',')
            channel_categories_list = []

            for category in category_id_list : 
                if str(category) in self.categories : 
                    category_label = self.categories[str(category)]
                    channel_categories_list = channel_categories_list + category_label
                else : 
#                     logger.info(f'ERROR {str(category)} not in categories ')
                    pass
            
            if output_batch[idx_i] == None:
                input_data = dict()
                input_data["framedata"] = framedata
                input_data["bbox"] = None
                input_data["scenario"] = scenario   
                input_data["data"] = None
                input_data["available"] = False
                res.append(input_data)
                continue

            for idx_j, output in enumerate(output_batch[idx_i]): 
                if isinstance(output, type(None)):
                    input_data = dict()
                    input_data["framedata"] = framedata
                    input_data["bbox"] = None
                    input_data["scenario"] = scenario   
                    input_data["data"] = None   
                    input_data["available"] = False
                    res.append(input_data)
                    continue
                
                label = int(output["label"]) + 1
                frame_count = framedata['meta']['source']['frame_count']

                if int(label) not in channel_categories_list : 
                    input_data = dict()
                    input_data["framedata"] = framedata
                    input_data["bbox"] = None
                    input_data["scenario"] = scenario   
                    input_data["data"] = None   
                    input_data["available"] = False
                    res.append(input_data)
#                     print(input_data['framedata']['meta']['source']['frame_count'] ,'not label')
                    continue

                input_data = dict()
                input_data["framedata"] = framedata
                input_data["bbox"] = output['bbox']
                input_data["scenario"] = scenario   
                input_data["data"] = {"score":output['score'], "label":str(label)}
                input_data["available"] = True
                
#                 print('##', input_data['framedata']['meta']['source']['frame_count'], input_data['bbox'], end='*')
                res.append(input_data)
        return res         
   
    
    def run_inference(self, input_data_batch, unavailable_routing_data_batch, reference_CM=None):
        
        
#         s1 = cuda.Stream(device="cuda")
#         s2 = cuda.Stream(device="cuda")
#         s3 = cuda.Stream(device="cuda")
    


        
        parsed_input_batch = self.parse_input(input_data_batch)
        
#         s1.wait_stream(cuda.current_stream())
#         with self.stream(self.s1):
        x,scale_list = self.preprocess(parsed_input_batch)
            

#         cuda.current_stream().wait_stream(self.s1)
#         with self.stream(self.s2):
        x = self.inference(x)
#         self.s3.wait_stream(cuda.current_stream())
        
        
        
#         with self.stream(self.s3):
        output_batch = self.postprocess(x,scale_list, class_agnostic=True)
        
#         cuda.current_stream().wait_stream(s1)
#         cuda.synchronize()
        output = self.parse_output(input_data_batch,output_batch,reference_CM)
        
        
            
            
#         t6 = time.time()
#         frame_time = (t6 - t1) / len(input_data_batch)
        
#         self.logger.debug(f'detect 1.parse_input : {t2 - t1}, 2.preprocess : {t3 - t2}, 3.inference : {t4 - t3}, 4.postprocess : {t5 - t4}, 5.parse_output : {t6 - t5},  6.total : {t6 - t1} 7. per_frame_time : {frame_time} 8. input_data_size : {len(input_data_batch)}')
#         self.logger.debug(f'________________________________________________________________________')
        
        return output, unavailable_routing_data_batch

def module_load(logger):     
    yd = DETECTOR(logger)
    return yd
    
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



# if __name__ == '__main__':
#     import logging
#     logger = logging.Logger('inference')
#     torch.cuda.init() #파이프 라인 필수 

#     ## 모델 생성
#     trt_engine_path = "/data/media_test/model_manager/engines/yoloxm_int8/yoloxm_int8_032.trt"
#     yd = module_load(logger)
#     yd.load(trt_engine_path)
    
#     # 샘플이미지 로드
#     img_path = "/data/media_test/model_manager/test/test_image.jpeg"
#     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#     img_path = "/data/media_test/model_manager/test/test_image.jpeg"
#     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#     img = cv2.resize(img, (300, 300))
#     img = torch.from_numpy(img).to(torch.device("cuda")) 
#     img = img.permute(2,0,1)

    
    
# #     img = torch.zeros(
# #         [3, 300, 300], 
# #         dtype=torch.float32,device=torch.device("cuda:0")
# #     ).fill_(144)
    


#     ## 더미데이터 생성
#     input_data = dict()
#     input_data["framedata"] = {"frame":img}
#     input_data["bbox"] = [0,0,img.shape[2],img.shape[1]]
#     input_data["scenario"] = "s"   
#     input_data["data"] = None   
    
#     ## 실제 데이터가 들어왔을때 배치만큼 리스트로 쌓여서 옴(4배치)
#     input_data_batch = [input_data for i in range(4)]
# #     print(f"input_data_batch : {input_data_batch}")
    
#     ## 추론 시작
#     output = yd.run_inference(input_data_batch)
#     print(f"output : {output}")
    
    
    

    


