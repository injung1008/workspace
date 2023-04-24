from interface_yoloxs import DETECTOR

from threading import Thread

import time
import torch
import torch.cuda as cuda
import torch.multiprocessing as mp
from collections import deque
import cv2


torch.cuda.init()


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
                
#             if cnt > 1200:
#                 break
                
                
                
                
            if cnt < 0:
                continue
            
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


def pre(dt, infer_queue, frame_list, stream, pre2post):
    st1 = cuda.Stream(device="cuda")
    pi_proc = dt.parse_input
    pre_proc = dt.preprocess
    start_time = time.time()
    print('start_time:', start_time)
    
    pi_result = pi_proc(frame_list[0])
    pre_result, scale_list = pre_proc(pi_result)
    infer_queue.put(pre_result)
    pre2post.append([frame_list[0], scale_list])
    time.sleep(30)
    for i in range(3):
        print(i+1, ' iter')
        for frame_batch in frame_list:
            with stream(st1):
                pi_result = pi_proc(frame_batch)
                pre_result, scale_list = pre_proc(pi_result)
                infer_queue.put(pre_result)
                pre2post.append([frame_batch, scale_list])
#             print('start_time:', time.time())
        time.sleep(20)
    while 1:
        time.sleep(10)
# class MM:
#     def __init__(self, dt):
#         self.infer_queue = mp.Queue()
#         self.post_queue = mp.Queue()
#         self.dt = dt
        
#     def inf(self):    
#         try:
#             weights = '/DATA_17/media_test/model_manager/engines/yoloxs_int8/yoloxs_best.trt'

#             infer_queue = self.infer_queue
#             post_queue = self.post_queue

#             self.dt.load(weights)
#             inference = self.dt.inference
#         #     time.sleep(40)
#             e1 = time.time()
#             while 1:
#                 if not infer_queue:
#                     time.sleep(0.0001)
#                     continue
#                 input_batch = infer_queue.get()
#         #         print('infer_queue get time : ',e-s)
#         #         input_batch = infer_queue.popleft()

#                 s1 = time.time()
#                 print('inference return time : ', s1-e1)
#                 infer_result = inference(input_batch)
#                 e1 = time.time()
#                 print('inference time : ',e1-s1, 'inference end time : ', time.time())
#         #         infer_result.share_memory_()
#                 post_queue.put(infer_result)
#         #         del input_batch
#         #         del infer_result
#         #         print(infer_result)
#         except Exception as e:
#             print(e)
            
            
            
            
        
def inf(dt, infer_queue, post_queue, stream):    
    weights = '/DATA_17/media_test/model_manager/engines/yoloxs_int8/yoloxs_best.trt'
    st2 = cuda.Stream(device="cuda")
#     dt.load(weights)
#     inference = dt.inference
    e1 = time.time()
    while 1:
        if infer_queue.empty():
            time.sleep(0.001)
            continue
#         with stream(st2):
#             s = time.time()
#             b_size = infer_queue.qsize()
#             input_data = infer_queue.get()
#             e = time.time()
#             print(f'infer_queue get time {b_size}->{infer_queue.qsize()} :{e-s}', end='//')

#             s1 = time.time()
#             print(f'inference return time :{s1-e1}', end='//')
#             infer_result = inference(input_data)
#             e1 = time.time()
#             print(f'inference time :{e1-s1}// inference end time :{time.time()}', end='//')

#             s2 = time.time()
#             print('infer_result',type(infer_result))
#             post_queue.put(infer_result)
#             e2 = time.time()
#             print(f'post_queue put time : {e2-s2}')
        
        with stream(st2):
            s = time.time()
            input_data = infer_queue.get()
            e = time.time()
            print(f'infer_queue get time :{e-s}', end='\n')

#             infer_result = inference(input_data)

#             post_queue.put(infer_result)
            post_queue.put(input_data)
            
def post(dt, post_queue, stream, pre2post):
    st3 = cuda.Stream(device="cuda")
    post_proc = dt.postprocess
    po_proc = dt.parse_output
    
    result_dict = {}
    
    count = 0
    while 1:
        if post_queue.empty():
            time.sleep(0.001)
            continue
        count += 1
        with stream(st3):
            input_data = post_queue.get()
            pre_batch = pre2post.popleft()
#             pre_data, scale_list = pre_batch
#     #         del post_batch
#     #         scale_list = [0.33333 for _ in range(len(post_batch))]

#             post_result = post_proc(input_data, scale_list)
#             po_result = po_proc(pre_data, post_result, None)
            
        for pr in po_result:
            fn = pr['framedata']['meta']['source']['frame_count']
            bbox = pr['bbox']
            if fn not in result_dict.keys():
                result_dict[fn] = []
            
            result_dict[fn].append(bbox)
#             if fn > 1197:
#                 time.sleep(10)
#                 for fnn in range(1200):
#                     if fnn not in result_dict or result_dict[fnn] is None:
#                         print(fnn)
#                 for fn, rd in result_dict.items():
# #                     for box in rd:
#                     print(f'## {fn} {rd}')
#         print('post_result : ', post_result)
#         print('end_time :', time.time(), count)
    
    
if __name__ == '__main__':
    
    dt = DETECTOR()
#     mm = MM(dt)
    
    batch_size = 16
#     video_path = '/DATA_17/ij/peopleNet_test/cocovid_sv_convert_30_.mp4'
    video_path = '/DATA_17/ij/peopleNet_test/0001_compressed.mp4'
    
    frame_list = proc(video_path, batch_size)
    print('frame load done')
    
    stream = cuda.stream 
    
#     s1 = cuda.Stream(device="cuda")
#     s2 = cuda.Stream(device="cuda")
#     s3 = cuda.Stream(device="cuda")
#     pi_proc = dt.parse_input
#     pre_proc = dt.preprocess
#     inference = dt.inference
#     post_proc = dt.postprocess
#     po_proc = dt.parse_output
    mp.set_start_method('spawn', force=True)
#     ctx = mp.get_context("spawn")
#     manager = mp.Manager()
#     infer_queue = manager.Queue()

#     infer_queue = ctx.Queue()
#     post_queue = ctx.Queue()
    infer_queue = mp.Queue()
    post_queue = mp.Queue()
    
    pre2post = deque()
    
    # infer_queue = deque()
    # post_queue = manager.Queue()
        
#     pre_p = ctx.Process(target=pre, args=(dt, infer_queue, frame_list,))
    pre_p = Thread(target=pre, kwargs={'dt':dt, 'infer_queue':infer_queue, 'frame_list':frame_list, 'stream':stream, 'pre2post':pre2post,})
    post_p = Thread(target=post, kwargs={'dt':dt, 'post_queue':post_queue, 'stream':stream,'pre2post':pre2post,})


    inf_p = mp.Process(target=inf, args=(dt, infer_queue, post_queue, stream,))
#     inf_p = mp.Process(target=mm.inf, args=())
    
    
    pre_p.start()
    post_p.start()
    inf_p.start()
    
    pre_p.join()
    post_p.join()
    inf_p.join()















