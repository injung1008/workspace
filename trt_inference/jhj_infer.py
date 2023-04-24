import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] ='0'
import sys
# sys.path.append('/DATA_17/hjjo/selftest/deep-high-resolution-net.pytorch/lib/core')
import numpy as np
import math
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import cv2 as cv
import torchvision
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import time
import hrnet_load_engine

# from utils.general import non_max_suppression

# ctx = cuda.Device(0).make_context()
# print(ctx)
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

      


def get_max_preds(batch_heatmaps):
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


def get_final_preds(batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if True:
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
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(
        center, scale, rot, output_size,
        shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv.getAffineTransform(np.float32(src), np.float32(dst))

    return trans
def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
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


#엔진 경로 설정해주기 
if True:
    hr_engine = hrnet_load_engine.Engine()
    batch_size = 1
    trt_engine_path = '/DATA_17/trt_test/engines/hrnet3/hrnet3_fp16_004.trt'
    img_path = '/DATA_17/hjjo/selftest/deep-high-resolution-net.pytorch/person_23_0_1.jpg'  
    image_bgr = cv.imread(img_path)
    image = image_bgr[:, :, [2, 1, 0]]
#     image = image_bgr
    model_input = cv.resize(image, (384,288))
#     model_input = torch.tensor(model_input).unsqueeze(0)
    rotation = 0
    center, scale = np.array([96.5, 215.5]), np.array([2.0203, 2.6937])
    # pose estimation transformation
    trans = get_affine_transform(center, scale, rotation, [288, 384])
    model_input = cv.warpAffine(
        image,
        trans,
        (288, 384),
        flags=cv.INTER_LINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
#     # pose estimation inference
    model_input = transform(model_input).unsqueeze(0)
    model_input = torch.tensor(model_input).clone().detach().cuda()
#     print(model_input, model_input.shape)

    engine ,context, stream = hr_engine.make_context(trt_engine_path, batch_size)

    #버퍼 할당해주기 
    hr_engine.allocate_buffers_all(batch_size, engine)

#     img_stack = img_process(img_path,batch_size)


    # input_data = torch.tensor(img_stack).cuda() #input 버퍼할당해 주지 않고 데이터를 바로 보낼때 
    result = hr_engine.do_inference_v2(context, model_input, None, output, stream) #결과 생성
    print('shape', result.shape) # (117504, )
    result = make_output(result, batch_size)
    print('after make_output', result, result.shape)
    
    
    print(center, scale)
    preds, _ = get_final_preds(
                result.clone().cpu().numpy(),
                np.asarray([center]),
                np.asarray([scale]))
    #         print(preds)
    print('predshape', preds.shape)
    print(preds)
    
    
    
    
    
    
    
    
    
    
    
    
    
    

