import time
import torch
import torch.multiprocessing as mp


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
        time.sleep(0.0001)
        
            
    
if __name__ == '__main__':
    mp.set_start_method('spawn')
    infer_queue = mp.Queue()
    inf_p = mp.Process(target=process1, args=(infer_queue,))
    inf_p.start()
    
    frame_list = []
    for i in range(500):
#         frame_list.append(torch.ones(3,720,1280).cuda(non_blocking=True))
        frame_list.append(torch.ones(3,1000,1000).cuda(non_blocking=True))
    print('frame
    for frame in frame_list:
        print('bf :  ', hex(frame.data_ptr()))
#         frame = frame.share_memory_()
        infer_queue.put(frame)
    
#     inf_p.join()


'''
bf :   0x7f5c24f9ec00
bf :   0x7f5c24fa8a00
bf :   0x7f5c24fb2800
proc : 0x7fc6d679ec00 // infer_queue get time :9.369850158691406e-05 // True
proc : 0x7fc6d67a8a00 // infer_queue get time :0.00011181831359863281 // True
proc : 0x7fc6d67b2800 // infer_queue get time :8.606910705566406e-05
0x7f9d0ec00000
0x7ff669000000
'''












