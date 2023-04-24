import time

import torch
import torch.multiprocessing as mp

def worker(q):
    while True:
        tensor = q.get()
        # tensor = q.get_nowait()
        if tensor is None:
            break
        
        if tensor.is_shared():
            print("after get tensor in shared memory.")
        else:
            print("after get tensor in global memory.")

        print('[{}]get tensor address : {}'.format(
            tensor[0], hex(tensor.data_ptr())))
        # print(tensor.mean())

if __name__ == "__main__":
    mp.set_start_method('fork')
    q = mp.Queue()
    process = mp.Process(target=worker, args=(q, ))
    process.start()

    for i in range(4):
        # tensor = torch.ones(181, 181) * i
        tensor = torch.ones(32768) * i
        tensor = tensor.cuda(non_blocking=True)

        if tensor.is_shared():
            print("Before put tensor in shared memory.")
        else:
            print("Before put tensor in global memory.")

        print('[{}]put tensor address : {}'.format(
            tensor[0], hex(tensor.data_ptr())))

        # q.put(tensor)
        q.put_nowait(tensor)
        

    q.put(None)
