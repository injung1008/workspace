import sys
import time

import torch
# from torch.multiprocessing import Process
# from torch.multiprocessing import Queue

import torch.multiprocessing as mp

def sample_data():
    t1 = time.time()
    t = torch.zeros([100, 3, 1080, 1920], dtype=torch.float,device=torch.device("cuda"))    
#     t.share_memory_()
    print(f"make time : {time.time() - t1}")
    return t

def test(q):
    print("started")
    while True:
        data = q.get()
        if data is None:
            time.sleep(0.001)
            continue
            
        print('Received data:', len(data),time.time())
        time.sleep(0.001)

# def test1(q):
    
#     data = q
#     print('Received data:',len(data),time.time())
#     time.sleep(0.001)


if __name__ == '__main__':    
#     mp.set_start_method('spawn', force=True)
    mp.set_start_method('spawn', force=True)
#     ctx = mp.get_context("spawn")
    q = mp.Queue()
#     q = ctx.Queue()
    
    p = mp.Process(target=test, args=(q,))
#     p = ctx.Process(target=test, args=(q,))
    p.start()    
    data = [sample_data() for i in range(3)]
    cnt = 0 
    while True :
        n = cnt % 3
        start = time.time()          
        q.put(data[n])
        end = time.time()
        print(f"data send start time : {start}")
        print(f"data send end time : {end}")
        print(f"data send diff time : {end-start}")
        time.sleep(5)
        cnt += 1
    p.join()

