from interface_people_hj_stream import PEOPLE_DETECTOR
from collections import deque
import time
import torch.multiprocessing as mp
from threading import Thread
import torch
import torch.cuda as cuda
import cv2
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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
#             if cnt > 600:
#                 break
                
            infe_frame = torch.from_numpy(frame).to(torch.device("cuda"),non_blocking=False)
            infe_frame = infe_frame.permute(2, 0, 1) 
            input_data = dict()
            input_data["framedata"] = {"frame":infe_frame}
            input_data["framedata"]['meta'] = {'source' : {'channel_id' : str(1_10), 'frame_count' : cnt }}
            input_data["bbox"] = [0,0,infe_frame.shape[2],infe_frame.shape[1]]
            input_data["scenario"] = "s"
            input_data["data"] = None   
            frame_list.append(input_data)
    return frame_list

def run_process(frame_queue):
    
    weights = '/DATA_17/ij/peopleNet_test/best_model_people.trt'
    
    number_of_context = 3
    
    pe = PEOPLE_DETECTOR()
    pe.load(weights)
    
    stream = cuda.stream
    
    model_queue = [deque() for _ in range(number_of_context)]
    sts = None
    result_pool = {}
    th_list = []
    spread_thread = Thread(target=frame_spread, kwargs={'frame_queue':frame_queue, 'model_queue':model_queue})
    th_list.append(spread_thread)
    for idx in range(number_of_context):
        th = Thread(target=run_stream, kwargs={'idx':idx, 'pe':pe, 'stream':stream, 'sts':sts, 
                                               'result_pool':result_pool, 'model_queue':model_queue[idx]})
        th_list.append(th)

    for th in th_list :
        th.start()
        print('th start')
        
    while 1:
        time.sleep(10)
        
def frame_spread(frame_queue, model_queue):
    
    number_of_context = 3
    spread_count = 0
    while True:
        if frame_queue.empty():
            for idx, mq in enumerate(model_queue):
                print(idx, 'queue size :', len(mq))
#             print('frame_queue empty')
            time.sleep(1)
            continue
        fq_s = time.time()
        frame_batch = frame_queue.get()
        fq_e = time.time()
        
#         print(f'frame_batch get : {fq_e-fq_s} // ')
        
        model_queue[spread_count].append(frame_batch)
#         print(f'{spread_count} model_queue append')

        spread_count = (spread_count+1) % number_of_context
        
        
def frame_send(frame_queue):
    
    video_path = '/DATA_17/ij/peopleNet_test/0001_compressed.mp4'
    frame_list = proc(video_path)
#     print('frema load done : ', len(frame_list))
    
    for frame_data in frame_list:
        frame_queue.put(frame_data)
#         print('frame_queue put')
       
    while 1:
        time.sleep(10)


def run_stream(idx, pe, stream, sts, result_pool, model_queue):
    
    st1 = cuda.Stream(device='cuda')
    time.sleep(30)
    inf_e = time.time()
    while 1:
        if not model_queue:
            time.sleep(0.0001)
            continue
            
        with stream(st1):
            mq_s = time.time()
            frame_data = model_queue.popleft()
            mq_e = time.time()
            
            
            rt_time = mq_e - inf_e
            res = pe.run_infer([frame_data], [], idx)
            inf_e = time.time()
            print(f'model_queue get [{idx}] : {mq_e-mq_s} // infer time : {inf_e-mq_e} // now time : {time.time()} // return time : {rt_time}')
        time.sleep(0.0001)
            
            
        
if __name__ == "__main__":


    
    mp.set_start_method('fork', force=True)
#     mp.set_start_method('spawn')
    
    frame_queue = mp.Queue()
    
    procs = mp.Process(target=run_process, args=(frame_queue,))
    frame_proc = mp.Process(target=frame_send, args=(frame_queue,))
        
    procs.start()
    frame_proc.start()
    procs.join()
    frame_proc.join()
        










