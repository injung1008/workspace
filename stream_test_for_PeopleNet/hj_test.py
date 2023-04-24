import time
import torch
import torch.multiprocessing as mp
import os

# torch.cuda.init()

def process1(infer_queue,):
    cnt = 0
    while 1:
        if infer_queue.empty():
            time.sleep(0.0001)
            continue
        s = time.time()
        input_data = infer_queue.get()
        e = time.time()
        print(f'proc : {hex(input_data.data_ptr())} // infer_queue get time :{cnt} {e-s} // {input_data.is_shared()}')

#         del input_data
        cnt += 1
#         time.sleep(0.0001)





if __name__ == '__main__':
    mp.set_start_method('spawn')
    infer_queue = mp.Queue()
    inf_p = mp.Process(target=process1, args=(infer_queue,))
    inf_p.start()

    frame_list = []
    for i in range(1000):
#         frame_list.append(torch.ones(3,720,1280).cuda(non_blocking=True))
        frame_list.append(torch.ones(3,500,500).cuda(non_blocking=True))
    print('frame load done')
    for frame in frame_list:
        print('bf :  ', hex(frame.data_ptr()))
#         frame = frame.share_memory_()
        infer_queue.put(frame)
