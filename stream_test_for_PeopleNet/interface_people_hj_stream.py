# import argparse
# import os
# import sys
# import numpy as np
# import tensorrt as trt
# import cv2
# import torchvision
# import torch
# import time
# import traceback
# import importlib
# import common_people as common
# from PIL import Image
# import matplotlib.pyplot as plt
# import torchvision.transforms.functional as TF

# # torch.cuda.init()

# @torch.jit.script
# def postprocess_jit(norm_data_x, norm_data_y, label_tensor, tensor_bbox,tensor_score):

#     tensor_x1 = torch.cat([tensor_bbox[:2040],tensor_bbox[8160:10200],tensor_bbox[16320:18360]],dim=0)
#     tensor_y1 = torch.cat([tensor_bbox[2040:4080],tensor_bbox[10200:12240],tensor_bbox[18360:20400]],dim=0)
#     tensor_x2 = torch.cat([tensor_bbox[4080:6120],tensor_bbox[12240:14280],tensor_bbox[20400:22440]],dim=0)
#     tensor_y2 = torch.cat([tensor_bbox[6120:8160],tensor_bbox[14280:16320],tensor_bbox[22440:]],dim=0)

#     tensor_x1 = (tensor_x1 - norm_data_x) * -35
#     tensor_y1 = (tensor_y1 - norm_data_y) * -35
#     tensor_x2 = (tensor_x2 + norm_data_x) * 35
#     tensor_y2 = (tensor_y2 + norm_data_y) * 35

#     total_bbox_tensor = torch.stack([tensor_x1,tensor_y1,tensor_x2,tensor_y2],dim=1)
#     score_tensor_mask = tensor_score >= 0.7

#     t_box = total_bbox_tensor[score_tensor_mask]
#     t_score = tensor_score[score_tensor_mask]
#     t_label = label_tensor[score_tensor_mask]


#     return t_box,t_score,t_label




# class PEOPLE_DETECTOR:

#     def __init__(self):  
# #         self.logger = logger

        
#         self.num_classes = list(range(3))
#         self.threshold = 0.7
#         self.nmsthre = 0.5


#         self.input_h = 544
#         self.input_w = 960

#         self.stride = 16
#         self.box_norm = 35.0
#         self.grid_h = 34
#         self.grid_w = 60
        
#         self.grid_calculator()

#     def load(self, weights):
#         self.weights = weights
#         self.Engine = common.Engine()      
#         self.Engine.make_context(self.weights)
#         self.batchsize = 1

#     def parse_input(self,input_data_batch):
#         res = []
#         for input_data in input_data_batch:
#             frame = input_data['framedata']['frame']
#             bbox = input_data['bbox']
#             cropped_img = common.getCropByFrame(frame,bbox)
#             res.append(cropped_img)
#         return res

#     def grid_calculator(self):

#         norm_data_x = torch.zeros([self.grid_h, self.grid_w], dtype=torch.float32, device=torch.device("cuda"))
#         norm_data_y = torch.zeros([self.grid_h, self.grid_w], dtype=torch.float32, device=torch.device("cuda"))

#         self.grid_h = int(self.input_h / self.stride)
#         self.grid_w = int(self.input_w / self.stride)
#         self.grid_size = self.grid_h * self.grid_w

#         for i in range(self.grid_h):
#             value = (i * self.stride + 0.5) / self.box_norm
#             norm_data_y[i,:] = value

#         for i in range(self.grid_w):
#             value = (i * self.stride + 0.5) / self.box_norm
#             norm_data_x[:,i] = value

            
#         norm_data_x = norm_data_x.view(-1,)
#         norm_data_y = norm_data_y.view(-1,)
        
#         self.norm_data_x = torch.cat([norm_data_x,norm_data_x,norm_data_x],-1)
#         self.norm_data_y = torch.cat([norm_data_y,norm_data_y,norm_data_y],-1)


#         label_c0 = torch.zeros([2040], dtype=torch.float32, device=torch.device("cuda")).fill_(0)
#         label_c1 = torch.zeros([2040], dtype=torch.float32, device=torch.device("cuda")).fill_(1)
#         label_c2 = torch.zeros([2040], dtype=torch.float32, device=torch.device("cuda")).fill_(2)
        
#         self.label_tensor = torch.cat([label_c0,label_c1,label_c2],-1)
        

#     def preprocess(self,frame_batch) : 

#         scale_list = []
#         for idx, frame in enumerate(frame_batch) :
#             _, h, w = frame.shape
#             permute = [2, 1, 0]
#             frame = frame[permute,:,:]
#             resized_img = torchvision.transforms.functional.resize(frame, (self.input_h, self.input_w)).float()
#             input_data = resized_img.div(255.0)
#             input_data = torch.ravel(input_data)
#             scale_list.append([h,w])
            

#         return input_data, scale_list

    
#     def inference(self,input_data) : 

#         output_data = self.Engine.do_inference_v2(input_data)
        

#         return output_data

#     def postprocess(self,outputs, scale_list):
#         """
#         Postprocesses the inference output
#         Args:
#             outputs (list of float): inference output
#             min_confidence (float): min confidence to accept detection
#             analysis_classes (list of int): indices of the classes to consider

#         Returns: list of list tuple: each element is a two list tuple (x, y) representing the corners of a bb
#         """

        
#         output_list = []
#         ori_h = scale_list[0][0]
#         ori_w = scale_list[0][1]
        
#         t_box,t_score,t_label = postprocess_jit(self.norm_data_x, self.norm_data_y, self.label_tensor, outputs[0],outputs[1])

        
#         start2 = time.time()
#         nms_out_index = torchvision.ops.batched_nms(
#             t_box,
#             t_score,
#             t_label,
#             self.nmsthre,
#         )
        
#         ############################################
#         result_box = t_box[nms_out_index].detach().cpu().numpy()
#         result_score = t_score[nms_out_index].detach().cpu().numpy()
#         result_label = t_label[nms_out_index].detach().cpu().numpy()

#         for idx, bbox in enumerate(result_box):
#             if isinstance(bbox, type(None)):
#                 output_list.append(None)
#                 continue
#             x1, y1, x2, y2 = map(int, bbox)
#             x1 = int((x1 * ori_w)/self.input_w)
#             y1 = int((y1 * ori_h)/self.input_h)
#             x2 = int((x2 * ori_w)/self.input_w)
#             y2 = int((y2 * ori_h)/self.input_h)
#             score = str(result_score[idx])
#             label = int(result_label[idx])
#             out = {"bbox":[x1, y1, x2, y2], "score":score, "label":label}
#             output_list.append(out)
#         ############################################

#         return output_list
    
    
    
#     def parse_output(self,input_data_batch,output_batch,reference_CM):
#         res = []
#         for idx_i, data in enumerate(input_data_batch): 
#             framedata = data['framedata']
#             scenario = data['scenario']
#             channel_id = framedata['meta']['source']['channel_id']
#             if output_batch == None:
#                 input_data = dict()
#                 input_data["framedata"] = framedata
#                 input_data["bbox"] = None
#                 input_data["scenario"] = scenario   
#                 input_data["data"] = None
#                 input_data["available"] = False
#                 res.append(input_data)
#                 continue
#             for idx_j, output in enumerate(output_batch): 

#                 if isinstance(output, type(None)):
#                     input_data = dict()
#                     input_data["framedata"] = framedata
#                     input_data["bbox"] = None
#                     input_data["scenario"] = scenario   
#                     input_data["data"] = None   
#                     input_data["available"] = False
#                     res.append(input_data)
#                     continue
                    
#                 label = int(output['label']) 
#                 frame_count = framedata['meta']['source']['frame_count']

#                 if int(label) != 0 : 
#                     input_data = dict()
#                     input_data["framedata"] = framedata
#                     input_data["bbox"] = None
#                     input_data["scenario"] = scenario   
#                     input_data["data"] = None   
#                     input_data["available"] = False
#                     res.append(input_data)
#                     continue

#                 input_data = dict()
#                 input_data["framedata"] = framedata
#                 input_data["bbox"] = output['bbox']
#                 input_data["scenario"] = scenario   
#                 input_data["data"] = {"score":output['score'], "label":str(output['label'])}
#                 input_data["available"] = True
#                 res.append(input_data)
#         return res  
        
#     def run_inference(self, input_data_batch, unavailable_routing_data_batch, reference_CM=None):

#         t1 = time.time()

#         parsed_input_batch = self.parse_input(input_data_batch)
#         t2 = time.time()
        
#         x,scale_list = self.preprocess(parsed_input_batch)
#         t3 = time.time()
        
#         x = self.inference(x)
#         t4 = time.time()
        
#         output_batch = self.postprocess(x,scale_list)
#         t5 = time.time()
        

#         output = self.parse_output(input_data_batch,output_batch,reference_CM)
#         t6 = time.time()
#         frame_time = (t6 - t1) / len(input_data_batch)
        
# #         print(f'[PEOPLE_DETECTOR] 1.parse_input : {t2 - t1}, 2.preprocess : {t3 - t2}, 3.inference : {t4 - t3}, 4.postprocess : {t5 - t4}, 5.parse_output : {t6 - t5},  6.total : {t6 - t1} 7. per_frame_time : {frame_time} 8. input_data_size : {len(input_data_batch)}')
# #         print(f'________________________________________________________________________')
        
#         return output, unavailable_routing_data_batch

# # def module_load(logger):     
# #     yd = PEOPLE_DETECTOR(logger)
# #     return yd

######################################################################################################################



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
import common_people_context as common
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch.cuda as cuda
# # torch.cuda.init()
# s1 = cuda.Stream(device="cuda")
# s2 = cuda.Stream(device="cuda")

class PEOPLE_DETECTOR:

    def __init__(self):  
#         self.logger = logger

        
        self.num_classes = list(range(3))
        self.threshold = 0.7
        self.nmsthre = 0.5


        self.input_h = 544
        self.input_w = 960

        self.stride = 16
        self.box_norm = 35.0
        self.grid_h = 34
        self.grid_w = 60
        
        self.grid_calculator()
        self.stream = cuda.stream
        
        self.s1 = cuda.Stream(device="cuda")
#         self.s2 = cuda.Stream(device="cuda")
#         self.s3 = cuda.Stream(device="cuda")

    def load(self, weights):
        self.weights = weights
        self.Engine = common.Engine()      
        self.Engine.make_context(self.weights)
        self.batchsize = 1

    def parse_input(self,input_data_batch):
        res = []
        for input_data in input_data_batch:
            frame = input_data['framedata']['frame']
            bbox = input_data['bbox']
            cropped_img = common.getCropByFrame(frame,bbox)
            res.append(cropped_img)
        return res
    

    def grid_calculator(self):

        norm_data_x = torch.zeros([self.grid_h, self.grid_w], dtype=torch.float32, device=torch.device("cuda"))
        norm_data_y = torch.zeros([self.grid_h, self.grid_w], dtype=torch.float32, device=torch.device("cuda"))

        self.grid_h = int(self.input_h / self.stride)
        self.grid_w = int(self.input_w / self.stride)
        self.grid_size = self.grid_h * self.grid_w

        for i in range(self.grid_h):
            value = (i * self.stride + 0.5) / self.box_norm
            norm_data_y[i,:] = value

        for i in range(self.grid_w):
            value = (i * self.stride + 0.5) / self.box_norm
            norm_data_x[:,i] = value

            
        norm_data_x = norm_data_x.view(-1,)
        norm_data_y = norm_data_y.view(-1,)
        
        self.norm_data_x = torch.cat([norm_data_x,norm_data_x,norm_data_x],-1)
        self.norm_data_y = torch.cat([norm_data_y,norm_data_y,norm_data_y],-1)


        label_c0 = torch.zeros([2040], dtype=torch.float32, device=torch.device("cuda")).fill_(0)
        label_c1 = torch.zeros([2040], dtype=torch.float32, device=torch.device("cuda")).fill_(1)
        label_c2 = torch.zeros([2040], dtype=torch.float32, device=torch.device("cuda")).fill_(2)
        
        self.label_tensor = torch.cat([label_c0,label_c1,label_c2],-1)
        




    def postprocess(self,outputs, scale_list):
        """
        Postprocesses the inference output
        Args:
            outputs (list of float): inference output
            min_confidence (float): min confidence to accept detection
            analysis_classes (list of int): indices of the classes to consider

        Returns: list of list tuple: each element is a two list tuple (x, y) representing the corners of a bb
        """

        
        output_list = []
        ori_h = scale_list[0][0]
        ori_w = scale_list[0][1]
        
        tensor_bbox = outputs[0]
        tensor_score = outputs[1]

        tensor_x1 = torch.cat([tensor_bbox[:2040],tensor_bbox[8160:10200],tensor_bbox[16320:18360]],dim=0)
        tensor_y1 = torch.cat([tensor_bbox[2040:4080],tensor_bbox[10200:12240],tensor_bbox[18360:20400]],dim=0)
        tensor_x2 = torch.cat([tensor_bbox[4080:6120],tensor_bbox[12240:14280],tensor_bbox[20400:22440]],dim=0)
        tensor_y2 = torch.cat([tensor_bbox[6120:8160],tensor_bbox[14280:16320],tensor_bbox[22440:]],dim=0)

        tensor_x1 = (tensor_x1 - self.norm_data_x) * -35
        tensor_y1 = (tensor_y1 - self.norm_data_y) * -35
        tensor_x2 = (tensor_x2 + self.norm_data_x) * 35
        tensor_y2 = (tensor_y2 + self.norm_data_y) * 35

        total_bbox_tensor = torch.stack([tensor_x1,tensor_y1,tensor_x2,tensor_y2],dim=1)
        score_tensor_mask = tensor_score >= self.threshold

        t_box = total_bbox_tensor[score_tensor_mask]
        t_score = tensor_score[score_tensor_mask]
        t_label = self.label_tensor[score_tensor_mask]
        
        start2 = time.time()
        nms_out_index = torchvision.ops.batched_nms(
            t_box,
            t_score,
            t_label,
            self.nmsthre,
        )
        
        ############################################
        result_box = t_box[nms_out_index].detach().cpu().numpy()
        result_score = t_score[nms_out_index].detach().cpu().numpy()
        result_label = t_label[nms_out_index].detach().cpu().numpy()

        for idx, bbox in enumerate(result_box):
            if isinstance(bbox, type(None)):
                output_list.append(None)
                continue
            x1, y1, x2, y2 = map(int, bbox)
            x1 = int((x1 * ori_w)/self.input_w)
            y1 = int((y1 * ori_h)/self.input_h)
            x2 = int((x2 * ori_w)/self.input_w)
            y2 = int((y2 * ori_h)/self.input_h)
            score = str(result_score[idx])
            label = int(result_label[idx])
            out = {"bbox":[x1, y1, x2, y2], "score":score, "label":label}
            output_list.append(out)
        ############################################

        return output_list
        


    def preprocess(self,frame_batch) : 

        scale_list = []
        for idx, frame in enumerate(frame_batch) :
            _, h, w = frame.shape
            permute = [2, 1, 0]
            frame = frame[permute,:,:]
            resized_img = torchvision.transforms.functional.resize(frame, (self.input_h, self.input_w)).float()
            input_data = resized_img.div(255.0)
            input_data = torch.ravel(input_data)
            scale_list.append([h,w])
            

        return input_data, scale_list

    
    def inference(self,input_data, idx) : 

        output_data = self.Engine.do_inference_v2(input_data, idx)
        

        return output_data

    
    def parse_output(self,input_data_batch,output_batch,reference_CM):
        res = []
        for idx_i, data in enumerate(input_data_batch): 
            framedata = data['framedata']
            scenario = data['scenario']
            channel_id = framedata['meta']['source']['channel_id']
            if output_batch == None:
                input_data = dict()
                input_data["framedata"] = framedata
                input_data["bbox"] = None
                input_data["scenario"] = scenario   
                input_data["data"] = None
                input_data["available"] = False
                res.append(input_data)
#                 print(input_data['framedata']['meta']['source']['frame_count'] ,'None')
                continue
            for idx_j, output in enumerate(output_batch): 

                if isinstance(output, type(None)):
                    input_data = dict()
                    input_data["framedata"] = framedata
                    input_data["bbox"] = None
                    input_data["scenario"] = scenario   
                    input_data["data"] = None   
                    input_data["available"] = False
                    res.append(input_data)
#                     print(input_data['framedata']['meta']['source']['frame_count'] ,'type None')
                    continue
                    
                label = int(output['label']) 
                frame_count = framedata['meta']['source']['frame_count']

                if int(label) != 0 : 
                    input_data = dict()
                    input_data["framedata"] = framedata
                    input_data["bbox"] = None
                    input_data["scenario"] = scenario   
                    input_data["data"] = None   
                    input_data["available"] = False
                    res.append(input_data)
#                     print(input_data['framedata']['meta']['source']['frame_count'] ,label, 'not label')
                    continue

                input_data = dict()
                input_data["framedata"] = framedata
                input_data["bbox"] = output['bbox']
                input_data["scenario"] = scenario   
                input_data["data"] = {"score":output['score'], "label":str(output['label'])}
                input_data["available"] = True
#                 print('##',frame_count, input_data["bbox"])
                res.append(input_data)
        return res  

#     def pre_queue(self):
        
        
#     def inf_queue(self):
        
        
        
#     def post_queue(self):
    
    def run_infer(self, input_data_batch, unavailable_routing_data_batch, idx, reference_CM=None):
        parsed_input_batch = self.parse_input(input_data_batch)
        x,scale_list = self.preprocess(parsed_input_batch)
        infer_result = self.inference(x, idx)
        output_batch = self.postprocess(infer_result,scale_list)
        output = self.parse_output(input_data_batch,output_batch,reference_CM)
        return output, unavailable_routing_data_batch
    
    def run_inference(self, input_data_batch, unavailable_routing_data_batch, reference_CM=None):
#         s1 = cuda.Stream(device="cuda")
#         s2 = cuda.Stream(device="cuda")
#         s3 = cuda.Stream(device="cuda")
        parsed_input_batch = self.parse_input(input_data_batch)
    
        with self.stream(self.s1):
            x,scale_list = self.preprocess(parsed_input_batch)
            
        cuda.current_stream().wait_stream(self.s1)
        
        infer_result = self.inference(x)
#         self.s3.wait_stream(cuda.current_stream())

#         with self.stream(self.s3):
        output_batch = self.postprocess(infer_result,scale_list)
        output = self.parse_output(input_data_batch,output_batch,reference_CM)
        return output, unavailable_routing_data_batch
    
#     def run_inference(self, input_data_batch, unavailable_routing_data_batch, reference_CM=None):
# #         s1 = cuda.Stream(device="cuda")
# #         s3 = cuda.Stream(device="cuda")
#         parsed_input_batch = self.parse_input(input_data_batch)
#         with self.stream(self.s1):
#             x,scale_list = self.preprocess(parsed_input_batch)
#         cuda.current_stream().wait_stream(self.s1)
#         x = self.inference(x)
#         self.s3.wait_stream(cuda.current_stream())
#         with self.stream(self.s3):
#             output_batch = self.postprocess(x,scale_list)
#         output = self.parse_output(input_data_batch,output_batch,reference_CM)
#         return output, unavailable_routing_data_batch
    
#     def run_inference(self, input_data_batch, unavailable_routing_data_batch, reference_CM=None):

#         parsed_input_batch = self.parse_input(input_data_batch)
        
#         s1 = cuda.Stream(device="cuda")
        
#         with cuda.stream(s1):
        
#             x,scale_list = self.preprocess(parsed_input_batch)
   
#             x = self.inference(x)
#             cuda.current_stream().wait_stream(s1)       
#             output_batch = self.postprocess(x,scale_list)

        

#         output = self.parse_output(input_data_batch,output_batch,reference_CM)
#         return output, unavailable_routing_data_batch

# def module_load(logger):     
#     yd = PEOPLE_DETECTOR(logger)
#     return yd