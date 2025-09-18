import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import mask as mask_util
from pathlib import Path


# 데모를 위해 제공된 JSON 문자열을 annotations.json 파일로 저장합니다.
json_string = ""
with open('annotations.json', 'w', encoding='utf-8') as f:
    f.write(json_string)

# 1. `json.load()`를 사용하여 파일을 읽습니다.
with open('/home/keti/ap_ws/Grounded-SAM-2/outputs/grounded_sam2_local_demo/grounded_sam2_all_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 딕셔너리 구조를 이용해 원하는 데이터에 접근합니다.
print('data len:',len(data))


iter=0
matching_info={}

for idx in data:# length : number of images
    # iter+=1
    # if iter > 3:
    #     break
    print(f"Image Path: {idx['image_path']}")
    img_n=Path(idx['image_path'])
    img_idname=img_n.stem
    img_id=int(img_idname)
    print(img_id)
    matching_info.setdefault(img_id, {})
    for i, annotation in enumerate(idx['annotations']):
        cls_name= annotation['class_name']
        if cls_name in ['black hole', 'contamination']:
            # print(f"Annotation {i+1}:")
            # print(f"  Class Name: {annotation['class_name']}")
            # print(f"  Bounding Box: {annotation['bbox']}")
            # print(f"  Score: {annotation['score']}")
            mask_info=annotation['segmentation']
            mask = mask_util.decode(mask_info)  # shape = (H, W, 1) or (H, W)
            mask = np.squeeze(mask)       # (H, W)로 변환
            matching_info[img_id][cls_name]=mask
       
print(matching_info)

for q in matching_info:
    print(f"Image ID: {q}")
    # for cls_name, mask in matching_info[q].items():
    #     print(f"  Class Name: {cls_name}, Mask Shape: {mask.shape}")
    #     plt.imshow(mask, cmap='gray')
    #     plt.title(f"Mask for {cls_name} in Image ID {q}")
    #     plt.show()
print(matching_info[35]['black hole'].shape)
print()
print(matching_info[35]['contamination'])
# ls=matching_info.items()
# print(ls)