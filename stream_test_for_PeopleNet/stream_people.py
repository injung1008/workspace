from interface_people_hj_stream import PEOPLE_DETECTOR
from collections import deque
import time
from threading import Thread
import torch
import torch.cuda as cuda
import cv2
import os

torch.cuda.init()


def proc(video_path):
    frame_list = []
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
                break
            if cnt < 0:
                continue
#             if cnt > 300:
#                 break
            infe_frame = torch.from_numpy(frame).to(torch.device("cuda"),non_blocking=True)
            infe_frame = infe_frame.permute(2, 0, 1) 
            input_data = dict()
            input_data["framedata"] = {"frame":infe_frame}
            input_data["framedata"]['meta'] = {'source' : {'channel_id' : str(1_10), 'frame_count' : cnt }}
            input_data["bbox"] = [0,0,infe_frame.shape[2],infe_frame.shape[1]]
            input_data["scenario"] = "s"   
            input_data["data"] = None   
            frame_list.append(input_data)
    return frame_list


def run_stream(idx, pe, stream, sts, frame_list, result_pool, model_queue):
    
#     st1 = cuda.Stream(device='cuda')
#     for frame_data in frame_list:
#     with stream(st1):
    while 1:
        if not model_queue[idx]:
            time.sleep(0.0001)
            continue
        frame_data = model_queue[idx].popleft()
        res = pe.run_infer([frame_data], [], idx)
        print(time.time())
#     print(res)
#         result_pool[idx] = len(res)
#         for rp in res:    
#             result_pool.append(rp)
#     print()
#     print(len(res))
        
if __name__ == "__main__":

    weights = '/DATA_17/ij/peopleNet_test/best_model_people.trt'

    pe = PEOPLE_DETECTOR()
    pe.load(weights)

    video_path = '/DATA_17/ij/peopleNet_test/0001_compressed.mp4'
    frame_list = proc(video_path)
    print('frema load done : ', len(frame_list))

    stream = cuda.stream
    # sts = dict()
    # for i in range(10):
    #     sts[i] = cuda.Stream(device="cuda")
    model_queue = [deque() for _ in range(5)]
    sts = None
    result_pool = {}
    th_list = []
    for idx in range(5):
        th = Thread(target=run_stream, kwargs={'idx':idx,'pe':pe, 'stream':stream, 'sts':sts, 
                                               'frame_list':frame_list, 'result_pool':result_pool, 'model_queue':model_queue})
        th_list.append(th)

    for th in th_list :
        th.start()
    for i in range(3):
        for idx, frame_data in enumerate(frame_list):
            stream_idx = idx % 5
            model_queue[stream_idx].append(frame_data)
    
                                          
#     th.join()
    













