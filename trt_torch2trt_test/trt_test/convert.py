import json

def json_convert(data_path):
    # load file
    with open(data_path, 'r') as file:
        json_dict = json.load(file)
    metas_list = json_dict[0]['metas']
    categories = []
    #딕셔너리 만들기 
    total_dict = {}
    for metas in metas_list:
        category = metas["category"]
        start_frame = metas["start_frame"]
        end_frame = metas["end_frame"]
        bbox_list = metas["bbox_list"]
        unique = metas["meta_id"]
        for idx, bbox in zip(range(start_frame,end_frame+1),bbox_list):
            bbox[0] = round(bbox[0])
            bbox[1] = round(bbox[1])
            bbox[2] = round(bbox[0] + bbox[2])
            bbox[3] = round(bbox[1] + bbox[3])  
            if not idx in total_dict:
                total_dict[idx] = dict()
            if not category in total_dict[idx]:
                total_dict[idx][category] = list()
            total_dict[idx][category].append(bbox)
        categories.append(category)
    categories = list(set(categories))




    return total_dict, categories

# data_path = '/home/jpark/workspace/aaaa/injung/json/gt.json'
# res = json_convert(data_path)


