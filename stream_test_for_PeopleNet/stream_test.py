from interface_yoloxs_stream import DETECTOR
import time
from threading import Thread
import torch
import torch.cuda as cuda
import cv2
import os

torch.cuda.init()


                
            
weights = '/DATA_17/media_test/model_manager/engines/yoloxs_int8/yoloxs_best.trt'

pe = DETECTOR()
pe.load(weights)

batch_size = 16
video_path = '/DATA_17/ij/peopleNet_test/0001_compressed.mp4'
def proc(video_path, batch_size):
    frame_list = []
    batch_stack = []
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    cnt = 0
    if cap.isOpened() :    
        while True:
            cnt+=1
            ret, frame = cap.read()
            if ret == False :
                if batch_stack:
                    frame_list.append(batch_stack)
                break
            if cnt < 0:
                continue
                
            if cnt > 300:
                break
            
            infe_frame = torch.from_numpy(frame).to(torch.device("cuda"),non_blocking=True)
            infe_frame = infe_frame.permute(2, 0, 1) 
            input_data = dict()
            input_data["framedata"] = {"frame":infe_frame}
            input_data["framedata"]['meta'] = {'source' : {'channel_id' : str(1_10), 'frame_count' : cnt }}
            input_data["bbox"] = [0,0,infe_frame.shape[2],infe_frame.shape[1]]
            input_data["scenario"] = "s"   
            input_data["data"] = None
            batch_stack.append(input_data)
            
            if cnt % batch_size == 0:
                frame_list.append(batch_stack)
                batch_stack = []
                
    return frame_list


def run_stream(idx, pe, stream, sts, frame_list, result_pool):
    
    st1 = cuda.Stream(device='cuda')
#     for frame_data in frame_list:
#     with stream(st1):
    res = pe.run_infer(frame_list[idx], [])
    print(res)
#         result_pool[idx] = len(res)
#         for rp in res:    
#             result_pool.append(rp)
#     print()
#     print(len(res))
        
frame_list = proc(video_path, batch_size)
print('frame load done, frame_list size : ', len(frame_list), '*', batch_size)

stream = cuda.stream
# sts = dict()
# for i in range(10):
#     sts[i] = cuda.Stream(device="cuda")

sts = None
result_pool = {}
th_list = []
for idx in range(1):
    th = Thread(target=run_stream, kwargs={'idx':idx,'pe':pe, 'stream':stream, 'sts':sts, 
                                           'frame_list':frame_list, 'result_pool':result_pool})
    th.start()
    th_list.append(th)

for th in th_list :
    th.join()
    













