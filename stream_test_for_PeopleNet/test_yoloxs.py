
from interface_yoloxs_stream import DETECTOR
import time
import cv2
import torch
torch.cuda.init()


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


                
weights = '/DATA_17/media_test/model_manager/engines/yoloxs_int8/yoloxs_best.trt'

pe = DETECTOR()
pe.load(weights)

frame_list = proc(video_path, batch_size)
print('frame load done, frame_list size : ', len(frame_list), '*', batch_size)

# from interface_dynamic import PEOPLE_DETECTOR

for i in range(10) : 
    start = time.time()
    for input_data in frame_list : 
        x, scale_list = pe.run_inference(input_data,[])

    print(f'end : {time.time() -start}')
raise
