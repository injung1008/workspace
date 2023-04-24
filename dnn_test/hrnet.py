import os
import sys
import numpy as np
import math
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import cv2 as cv
import torchvision
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch
import common as common
import time

class HRNET_POSE:
    def __init__(self):
        self.score_thresh = 0.65
        self.start_img_stack = 1
        self.scale_list = None
        self.pad_list = None
        self.input_w = 288
        self.input_h = 384
    
    def load(self, trt_engine_path):
        self.Engine = common.Engine()      
        self.Engine.make_context(trt_engine_path)

        
    def preprocess(self, imgs):
        tensor_batch = torch.zeros([len(imgs), 3, self.input_h, self.input_w], dtype=torch.float32, device=torch.device("cuda")).fill_(144)
        scale_list = list()
        pad_list = list()
        t_h, t_w = 374, 278
        
        for idx, img in enumerate(imgs):
            
            
            img = torch.from_numpy(img).to(torch.device("cuda")).permute(2, 0, 1).div(255)

            _, h, w = img.shape
            
            # resize
            rate_w = t_w / w
            rate_h = t_h / h
            rate = rate_h if rate_w > rate_h else rate_w
            img = transforms.functional.resize(img, size=(int(h*rate), int(w*rate)))
       
            
            # pad
            _, r_h, r_w = img.shape
            left_pad = int((self.input_w - r_w) / 2)
            right_pad = self.input_w - (r_w + left_pad)
            top_pad = int((self.input_h - r_h) / 2)
            bottom_pad = self.input_h - (r_h + top_pad)
            img = transforms.functional.pad(img, padding=[left_pad, top_pad, right_pad, bottom_pad])

            
            # normalize
            img = transforms.functional.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

            # put 
            tensor_batch[idx, :, :self.input_h, :self.input_w] = img
            scale_list.append(rate)
            pad_list.append([left_pad, top_pad, right_pad, bottom_pad])

        self.scale_list = scale_list
        self.pad_list = pad_list
        return tensor_batch

        
    def postprocess(self, model_output):
        
        model_output = np.array(model_output.cpu())
        
        preds, maxvals = self.get_final_preds(
                    model_output,
                    np.asarray(self.scale_list),
                    np.asarray(self.pad_list))
        
        frontback = self.checkFrontBack(preds)
        
        return preds, maxvals, frontback

    def get_max_preds(self, batch_heatmaps):
        '''
        get predictions from score maps
        heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
        '''
        assert isinstance(batch_heatmaps, np.ndarray), \
            'batch_heatmaps should be numpy.ndarray'
        assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

        batch_size = batch_heatmaps.shape[0]
        num_joints = batch_heatmaps.shape[1]
        width = batch_heatmaps.shape[3]
        heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask
        return preds, maxvals


    def get_final_preds(self, batch_heatmaps, scale_list, pad_list):    
        coords, maxvals = self.get_max_preds(batch_heatmaps)
        heatmap_height = batch_heatmaps.shape[2]
        heatmap_width = batch_heatmaps.shape[3]
        # post-processing
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],
                            hm[py+1][px]-hm[py-1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25
        preds = coords.copy()
        for idx in range(len(preds)):
            preds[idx] = preds[idx,:,:] * 4
            preds[idx,:,0] = preds[idx,:,0] - pad_list[idx,0]
            preds[idx,:,1] = preds[idx,:,1] - pad_list[idx,1]
            preds[idx] = preds[idx,:,:] / scale_list[idx]
        return preds, maxvals


    def checkFrontBack(self, preds):        
        temp_arr = np.array([])
        temp_arr = np.append(temp_arr,np.where(preds[:,5,:1] - preds[:,6,:1] >= 0,True,False).reshape(-1))
        temp_arr = np.append(temp_arr,np.where(preds[:,7,:1] - preds[:,8,:1] >= 0,True,False).reshape(-1))
        temp_arr = np.append(temp_arr,np.where(preds[:,9,:1] - preds[:,10,:1] >= 0,True,False).reshape(-1))
        temp_arr = np.append(temp_arr,np.where(preds[:,11,:1] - preds[:,12,:1] >= 0,True,False).reshape(-1))
        temp_arr = np.append(temp_arr,np.where(preds[:,13,:1] - preds[:,14,:1] >= 0,True,False).reshape(-1))
        temp_arr = np.append(temp_arr,np.where(preds[:,15,:1] - preds[:,16,:1] >= 0,True,False).reshape(-1))
        new_arr = np.reshape(temp_arr,(6 ,int(len(temp_arr) / 6)))
        new_arr = np.transpose(new_arr)
        return np.where(np.count_nonzero(new_arr,axis=1) >= 3,"FRONT","BACK")
    
    def get_dir(self, src_point, rot_rad):
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result


    def get_affine_transform(self, center, scale, rot, output_size,
            shift=np.array([0, 0], dtype=np.float32), inv=0
    ):
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale])

        scale_tmp = scale * 200.0
        src_w = scale_tmp[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = self.get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift
        src[1, :] = center + src_dir + scale_tmp * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        src[2:, :] = self.get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = self.get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv.getAffineTransform(np.float32(src), np.float32(dst))

        return trans
    def transform_preds(self, coords, center, scale, output_size):
        target_coords = np.zeros(coords.shape)
        trans = self.get_affine_transform(center, scale, 0, output_size, inv=1)
        for p in range(coords.shape[0]):
            target_coords[p, 0:2] = self.affine_transform(coords[p, 0:2], trans)
        return target_coords

    def affine_transform(self, pt, t):
        new_pt = np.array([pt[0], pt[1], 1.]).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]


    def get_3rd_point(self, a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    def box_to_center_scale(self, box, model_image_width, model_image_height):
        center = np.zeros((2), dtype=np.float32)
        bottom_left_corner = box[0]
        top_right_corner = box[1]
        box_width = top_right_corner[0]-bottom_left_corner[0]
        box_height = top_right_corner[1]-bottom_left_corner[1]
        bottom_left_x = bottom_left_corner[0]
        bottom_left_y = bottom_left_corner[1]
        center[0] = bottom_left_x + box_width * 0.5
        center[1] = bottom_left_y + box_height * 0.5

        aspect_ratio = model_image_width * 1.0 / model_image_height
        pixel_std = 200

        if box_width > aspect_ratio * box_height:
            box_height = box_width * 1.0 / aspect_ratio
        elif box_width < aspect_ratio * box_height:
            box_width = box_height * aspect_ratio
        scale = np.array(
            [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    
    def inference(self,input_data) : 
        result = self.Engine.do_inference_v2(input_data)
#         print(result)
        print(result.shape)
        return result

        
def module_load():
    hrn = HRNET_POSE()
    return hrn

#SAMPLE 
if __name__ == '__main__':
    torch.cuda.init()
    trt_engine_path = '/DATA_17/trt_test/engines/hrnet_test/hrnet3_fp16_006.trt'
    img_path = '/DATA_17/hjjo/selftest/deep-high-resolution-net.pytorch/person_23_0_1.jpg'  

    img = cv.imread(img_path)
    img_list = [img for _ in range(6)]
    hrn = HRNET_POSE()
    hrn.load(trt_engine_path)
    
    input_data = hrn.preprocess(img_list)

    
    output_data = hrn.inference(input_data)

    
    pred, maxvals, frontback = hrn.postprocess(output_data)
    
    print('pred', pred, maxvals, frontback)












