from interface_yoloxs import DETECTOR

from threading import Thread

import time
import torch
import torch.cuda as cuda
import torch.multiprocessing as mp
from collections import deque
import cv2


torch.cuda.init()



        
def process1():    
    cnt = 0
    while 1:
        
        s = time.time()
        a = torch.ones([3,1000,1000]).to(torch.device('cuda'), non_blocking=True)
        e = time.time()
        print(e-s)
            
        cnt += 1
        if cnt > 1200 :
            break
    
    
if __name__ == '__main__':
    
    mp.set_start_method('spawn', force=True)
    inf_p = mp.Process(target=process1)
    inf_p.start()

    
    inf_p.join()















