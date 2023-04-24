import sys
import torch
import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

sys.path.insert(0,"/DATA_17/hjjo/selftest/yolov5")
from models.common import DetectMultiBackend

    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5l.pt', help='model path(s)')
    parser.add_argument('--onnx_path', type=str, default='yolov5l.onnx', help='onnx path(s)')
    parser.add_argument('--size', type=int, default='640', help='input shape(width or height)')
    parser.add_argument('--onnx-batch', type=int, default='1', help='onnx convert batch size')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--yaml', default='coco128.yaml', help='.yaml file path')
    parser.add_argument('--half', default=False, action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', default=False, action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    return opt


def pth2onnx(opt):
    
    # device = select_device('3')
    device = torch.device('cuda:0')

    # model load
    model = DetectMultiBackend(opt.weights, device=device, dnn=opt.dnn, data=opt.yaml, fp16=opt.half)
    print('- PTH Model Loaded -')

    # input setting
    input = torch.randn(opt.onnx_batch, 3, opt.size, opt.size).cuda()

    # inference model
    output = model(input)

    # pytorch to onnx (dynamic shape)
    torch.onnx.export(
        model,
        input, 
        opt.onnx_path,
        verbose=True, 
        opset_version=11,
        input_names = ['input'],
        output_names = ['output'],
        dynamic_axes = {'input' : {0 : 'batch_size'},
                        'output' : {0 : 'batch_size'}}
    )
    print('- ONNX Convert Complete -')

    
    

if __name__ == '__main__':
    # setting parameter
    opt = parse_opt()
    
    # run pth to onnx, onnx to trt converting
    pth2onnx(opt)





