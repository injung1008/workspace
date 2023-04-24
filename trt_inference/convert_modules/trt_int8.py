import cv2 as cv
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import argparse
import numpy as np

import sys, os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


from calibrator import EntropyCalibrator
import common

TRT_LOGGER = trt.Logger()

def preproc2(img, input_size, swap=(2, 0, 1)):
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
    return padded_img, r


def getAllFilePath(root_dir,extensions): 
    img_path_list = []
    for (root, dirs, files) in os.walk(root_dir):
        if len(files) > 0:
            for file_name in files:
                if os.path.splitext(file_name)[1] in extensions:
                    img_path = root + '/' + file_name
                    img_path = img_path.replace('\\', '/')      
                    img_path_list.append(img_path)
    return img_path_list        


def convertDataset2(dataset_path_list, opt):
    res_list = []
    for path in dataset_path_list :
        img = cv.imread(path)
        padded_img,r = preproc2(img,[opt.wsize,opt.hsize])
        res_list.append(padded_img)
        
    res_list = np.array(res_list) 
#     res_list = res_list.dtype(np.float32)
#     res_list = res_list / 255
    return res_list
        
    

def get_static_cache(opt, train_dataset=None):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():             
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
            config.max_workspace_size = 1 << opt.workspace_size # 256MiB

            calib = EntropyCalibrator(train_dataset, cache_file=opt.cache_path,batch_size=1)
    
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calib
            
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(opt.onnx_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(opt.onnx_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(opt.onnx_path))
            with open(opt.onnx_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 3, opt.wsize,opt.hsize]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(opt.onnx_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(opt.trt_path, "wb") as f:
                f.write(plan)
            return engine
        

    if os.path.exists(opt.trt_path):
        print("Reading engine from file {}".format(opt.trt_path))
        with open(opt.trt_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()        

    
def get_dynamic_engine(opt, train_dataset=None):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():             
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
            config.max_workspace_size = 1 << 32 # 256MiB
#             config.set_flag(trt.BuilderFlag.FP16)
            

            calib = EntropyCalibrator(train_dataset, cache_file=opt.cache_path,batch_size=1)
    
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calib
            
            #### dynamic shape
            profile = builder.create_optimization_profile()
            profile.set_shape("input", [1,3,opt.wsize,opt.hsize], [opt.trt_batch,3,opt.wsize,opt.hsize], [opt.trt_batch,3,opt.wsize,opt.hsize]) 
            config.add_optimization_profile(profile)            
            ####
            
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(opt.onnx_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(opt.onnx_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(opt.onnx_path))
            with open(opt.onnx_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(opt.onnx_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(opt.trt_path, "wb") as f:
                f.write(plan)
            return engine
        

    if os.path.exists(opt.trt_path):
        print("Reading engine from file {}".format(opt.trt_path))
        with open(opt.trt_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()  

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path', type=str, default='../make_trt/yolox_m.onnx', help='onnx path(s)')
    parser.add_argument('--trt_path', type=str, default="yolox_m_int8.trt", help='trt path(s)')
    parser.add_argument('--cache_path', type=str, default="yolox_m_calibration.cache", help='cache path')
    parser.add_argument('--dataset', type=str, default="/DATA_17/hjjo/YOLOX_pruning/YOLOX/datasets/COCO/val2017", help='dataset path')
    parser.add_argument('--datatype', type=str, default='int8', help='data type, fp16 or int8')
    parser.add_argument('--wsize', type=int, default='640', help='input shape(width or height)')
    parser.add_argument('--hsize', type=int, default='640', help='input shape(width or height)')
    parser.add_argument('--onnx-batch', type=int, default='1', help='onnx convert batch size')
    parser.add_argument('--trt_batch', type=int, default='128', help='TensorRT convert batch size')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workspace_size', type=int, default='32', help='workspace size')
    opt = parser.parse_args()
    return opt
    

def main():
    
    # setting parameter
    opt = parse_opt()
    
    # dataset load
    dataset_path_list = getAllFilePath(opt.dataset,[".jpg"])
    dataset_path_list = dataset_path_list[:2000]
    train_dataset1 = convertDataset2(dataset_path_list, opt)
   
    # static engine & cache file create
    with get_static_cache(opt, train_dataset=train_dataset1) as engine, engine.create_execution_context() as context:
        pass
    
    print('Saved static engine & cache file')
    
    # remove static engine file
    os.remove(opt.trt_path)
    print('Static engine file removed')
    
    # dynamic engine create
    with get_dynamic_engine(opt) as engine, engine.create_execution_context() as context:
        pass
    print("Dynamic engine file saved : ", opt.trt_path)
        
        
if __name__ == '__main__':
    main()
    
