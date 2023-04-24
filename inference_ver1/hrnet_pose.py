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


class HRNET_POSE:
    def __init__(self):
        self.score_thresh = 0.65
        self.center = None
        self.scale = None
        self.start_img_stack = 1
    
    def load(self, trt_engine_path):
        self.Engine = common.Engine()      
        self.Engine.make_context(trt_engine_path)
        _, self.input_w, self.input_h = self.Engine.allocate_buffers_all()


        
    def preproc(self, img):
        image = img[:, :, [2, 1, 0]]
        img_tensor = torch.from_numpy(img/255.).permute(2,0,1).float()
        box = [(0,0), (img_tensor.shape[2], img_tensor.shape[1])]
        rotation = 0
        self.center, self.scale = self.box_to_center_scale(box, img_tensor.shape[2], img_tensor.shape[1])
        trans = self.get_affine_transform(self.center, self.scale, rotation, [self.input_w,self.input_h])
        
        model_input = cv.warpAffine(
            image,
            trans,
            (self.input_w,self.input_h),
            flags=cv.INTER_LINEAR)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        
        model_input = transform(model_input).unsqueeze(0)
        model_input = torch.tensor(model_input).cuda()

        return model_input
    
    def img_process(self, img):
        img = self.preproc(img)
        img_list = [img for _ in range(self.start_img_stack)]
        img_stack = torch.cat(img_list, dim=0)
    
        return img_stack
        
    def make_output(self, model_output):
        result = np.reshape(model_output,(-1,17,96,72))
        
        result = result[0:self.img_batch]
        
        preds, maxvals = self.get_final_preds(
                    result,
                    np.asarray([self.center]),
                    np.asarray([self.scale]))
        
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


    def get_final_preds(self, batch_heatmaps, center, scale):
        coords, maxvals = self.get_max_preds(batch_heatmaps)

        heatmap_height = batch_heatmaps.shape[2]
        heatmap_width = batch_heatmaps.shape[3]

        # post-processingz
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

        # Transform back
        for i in range(coords.shape[0]):
            preds[i] = self.transform_preds(
                coords[i], center[0], scale[0], [heatmap_width, heatmap_height]
            )

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
        result, self.img_batch = self.Engine.do_inference_v2(input_data)
        return result
    
    def release(self):
        self.Engine.flush()
        
        

#SAMPLE 
if __name__ == '__main__':
    trt_engine_path = '/DATA_17/trt_test/engines/hrnet3/hrnet3_fp16_004.trt'
    img_path = '/DATA_17/hjjo/selftest/deep-high-resolution-net.pytorch/person_23_0_1.jpg'  

    img = cv.imread(img_path)
    hrn = HRNET_POSE()
    hrn.load(trt_engine_path)
    
    model_input = hrn.img_process(img)
    print('model_input.shape', model_input.shape)
    
    model_output = hrn.inference(model_input)
    print('model_output', model_output)
    
    pred, maxvals, frontback = hrn.make_output(model_output)
    print('pred', pred, maxvals, frontback)

    hrn.release()










