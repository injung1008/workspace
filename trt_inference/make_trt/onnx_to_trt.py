# import cv2 as cv
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch 
import onnx
# import onnxsim
import numpy as np
import random

import sys, os


TRT_LOGGER = trt.Logger()
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:            
            config.max_workspace_size = 1 << 32 # 256MiB
            config.set_flag(trt.BuilderFlag.FP16) #fp16이라고 지정 
#             config.max_workspace_size = 1 << 28 # 256MiB
            builder.max_batch_size = 16

            ### dynamic shape
            profile = builder.create_optimization_profile()
#             profile.set_shape("input", [1,3,256,128], [64,3,256,128], [64,3,256,128]) #배치, 채널, 세로(h), 가로(w)
            config.add_optimization_profile(profile)
            ###

            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
#                 print(model.read())
                print('Beginning ONNX file parsing')
#                 model = onnx.load(onnx_file_path)
#                 print(onnx.checker.check_model(model)

                
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print('record')
                        print (parser.get_error(error))
                    print('error in 55')
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
#             network.get_input(0).shape = [1, 3, 640, 640]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            
#             engine = builder.build_engine(network, config)
#             print(bytearray(engine.serialize()))
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine


    if os.path.exists(engine_file_path):
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def main():
    onnx_file_path = '/DATA_17/trt_test/onnx/yolor_car_brand.onnx'
    engine_file_path = '/DATA_17/ij/yolor_car_brand.trt'

    trt_outputs = []
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        pass
    print("done")


if __name__ == '__main__':
    main()
