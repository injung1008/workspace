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
import common_dynamic as common
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

# torch.cuda.init()


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
#         self.norm_data_x = norm_data_x
#         self.norm_data_y = norm_data_y    
        self.norm_data_x = torch.stack((norm_data_x,norm_data_x,norm_data_x))
        self.norm_data_y = torch.stack((norm_data_y,norm_data_y,norm_data_y))

    


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

#         tensor_x1 = torch.stack((outputs[0][0],outputs[0][4],outputs[0][8]))
#         tensor_y1 = torch.stack((outputs[0][1],outputs[0][5],outputs[0][9]))
#         tensor_x2 = torch.stack((outputs[0][2],outputs[0][6],outputs[0][10]))
#         tensor_y2 = torch.stack((outputs[0][3],outputs[0][7],outputs[0][11]))

        tensor_x1 = torch.cat([outputs[0][0],outputs[0][4],outputs[0][8]],0).view(-1,self.grid_h,self.grid_w)
        tensor_y1 = torch.cat([outputs[0][1],outputs[0][5],outputs[0][9]],0).view(-1,self.grid_h,self.grid_w)
        tensor_x2 = torch.cat([outputs[0][2],outputs[0][6],outputs[0][10]],0).view(-1,self.grid_h,self.grid_w)
        tensor_y2 = torch.cat([outputs[0][3],outputs[0][7],outputs[0][11]],0).view(-1,self.grid_h,self.grid_w)
        
        tensor_x1 = (tensor_x1 - self.norm_data_x) * -35
        tensor_y1 = (tensor_y1 - self.norm_data_y) * -35
        tensor_x2 = (tensor_x2 + self.norm_data_x) * 35
        tensor_y2 = (tensor_y2 + self.norm_data_y) * 35



        
        score_tensor = outputs[1] >= self.threshold
        
        for label in self.num_classes : 
#             ii = label * 4
            
#             tensor_x1 = (outputs[0][ii] - self.norm_data_x) * -35
#             tensor_y1 = (outputs[0][ii + 1] - self.norm_data_y) * -35
#             tensor_x2 = (outputs[0][ii + 2] + self.norm_data_x) * 35
#             tensor_y2 = (outputs[0][ii + 3] + self.norm_data_y) * 35

#             new_x1 = tensor_x1[score_tensor[label]]
#             new_y1 = tensor_y1[score_tensor[label]]
#             new_x2 = tensor_x2[score_tensor[label]]
#             new_y2 = tensor_y2[score_tensor[label]]

            new_x1 = tensor_x1[label][score_tensor[label]]
            new_y1 = tensor_y1[label][score_tensor[label]]
            new_x2 = tensor_x2[label][score_tensor[label]]
            new_y2 = tensor_y2[label][score_tensor[label]]


            box = torch.stack((new_x1,new_y1,new_x2,new_y2),1)
            score = outputs[1][label][score_tensor[label]]

            nms_out_index = torchvision.ops.nms(
                box,
                score,
                self.nmsthre,
            )
            output_list.append(None)
            
            ############################################
            result_box = box[nms_out_index].detach().cpu().numpy()
            result_score = score[nms_out_index].detach().cpu().numpy()

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
                out = {"bbox":[x1, y1, x2, y2], "score":score, "label":label}
                output_list.append(out)
            ############################################

        return output_list
        


    def preprocess(self,frame_batch) : 

#         input_data = torch.zeros([3, self.input_h, self.input_w], dtype=torch.float32, device=torch.device("cuda")).fill_(144)
        scale_list = []
        
        for idx, frame in enumerate(frame_batch) :
            _, h, w = frame.shape
            permute = [2, 1, 0]
            frame = frame[permute,:,:]
            resized_img = torchvision.transforms.functional.resize(frame, (self.input_h, self.input_w)).float()
            input_data = resized_img.div(255.0)
#             input_data[:,:self.input_h,:self.input_w] = resized_img 
            scale_list.append([h,w])
        return input_data, scale_list

    
    def inference(self,input_data) : 
        output_data = self.Engine.do_inference_v2(input_data)
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
                    continue

                input_data = dict()
                input_data["framedata"] = framedata
                input_data["bbox"] = output['bbox']
                input_data["scenario"] = scenario   
                input_data["data"] = {"score":output['score'], "label":str(output['label'])}
                input_data["available"] = True
                res.append(input_data)
        return res  
        
    def run_inference(self, input_data_batch, unavailable_routing_data_batch, reference_CM=None):

        t1 = time.time()

        parsed_input_batch = self.parse_input(input_data_batch)
        t2 = time.time()
        
        x,scale_list = self.preprocess(parsed_input_batch)
        t3 = time.time()
        
        x = self.inference(x)
        t4 = time.time()
        
        output_batch = self.postprocess(x,scale_list)
        t5 = time.time()
        

        output = self.parse_output(input_data_batch,output_batch,reference_CM)
        t6 = time.time()
        frame_time = (t6 - t1) / len(input_data_batch)
        
#         print(f'[PEOPLE_DETECTOR] 1.parse_input : {t2 - t1}, 2.preprocess : {t3 - t2}, 3.inference : {t4 - t3}, 4.postprocess : {t5 - t4}, 5.parse_output : {t6 - t5},  6.total : {t6 - t1} 7. per_frame_time : {frame_time} 8. input_data_size : {len(input_data_batch)}')
#         print(f'________________________________________________________________________')
        
        return output, unavailable_routing_data_batch

# def module_load(logger):     
#     yd = PEOPLE_DETECTOR(logger)
#     return yd