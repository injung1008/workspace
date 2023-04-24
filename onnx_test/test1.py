## pth => onnx 변환
import torch
import numpy as np


model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
ckpt = torch.load("resNet50_2_4_NHWC.pth", map_location="cpu")
model.load_state_dict(ckpt["state_dict"])
model.eval()
model.cuda()
batch_size = 1
input = np.ones(shape=(batch_size,3,224,224)).astype(np.float32)
input = torch.from_numpy(input).cuda()


## pytorch to onnx 
torch.onnx.export(
    model,                                # model being run
    input,    # model input (or a tuple for multiple inputs)
    "resNet50_2_4_NHWC.onnx", # where to save the model (can be a file or file-like object)
    verbose=True, 
    opset_version=11,
    input_names = ['input'],              # the model's input names
    output_names = ['output'],
    dynamic_axes = {'input' : {0 : 'batch_size'},
                    'output' : {0 : 'batch_size'}}
)            # the model's output names


