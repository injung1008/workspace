import sys
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

TRT_LOGGER = trt.Logger()

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

import common
# from trt_fp16 import get_engine, preprocess

    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path', type=str, default='yolov5l.onnx', help='onnx path(s)')
    parser.add_argument('--trt_path', type=str, default='yolov5l.trt', help='trt path(s)')
    parser.add_argument('--datatype', type=str, default='fp16', help='data type, fp16 or int8')
    parser.add_argument('--wsize', type=int, default='640', help='input shape(width or height)')
    parser.add_argument('--hsize', type=int, default='640', help='input shape(width or height)')
    parser.add_argument('--onnx-batch', type=int, default='1', help='onnx convert batch size')
    parser.add_argument('--trt_batch', type=int, default='128', help='TensorRT convert batch size')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workspace_size', type=int, default='32', help='workspace size')
    opt = parser.parse_args()
    return opt

def get_engine(opt):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
            config.max_workspace_size = 1 << opt.workspace_size # 256MiB
            config.set_flag(trt.BuilderFlag.FP16)
#             config.max_workspace_size = 1 << 28 # 256MiB
            builder.max_batch_size = 1

            #### dynamic shape
            profile = builder.create_optimization_profile()
            profile.set_shape("input", [1,3,opt.wsize,opt.hsize], 
                                       [opt.trt_batch,3,opt.wsize,opt.hsize], 
                                       [opt.trt_batch,3,opt.wsize,opt.hsize])
            config.add_optimization_profile(profile)
            ####

            # Parse model file
            if not os.path.exists(opt.onnx_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(opt.onnx_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(opt.onnx_path))
            with open(opt.onnx_path, 'rb') as model:
#                 print(model.read())
                print('Beginning ONNX file parsing')
#                 model = onnx.load(onnx_file_path)
#                 print(onnx.checker.check_model(model)

                
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print('record')
                        print (parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
#             network.get_input(0).shape = [1, 3, 640, 640]
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
    
def onnx2trt(opt):
    # onnx to trt_fp16
    with get_engine(opt) as engine, engine.create_execution_context() as context:
        pass
    print('- TensorRT Convert Complete -')
    
    

if __name__ == '__main__':
    # setting parameter
    opt = parse_opt()
    
    # run onnx to trt converting
    onnx2trt(opt)




