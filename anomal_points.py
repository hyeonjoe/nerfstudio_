import os
from PIL import Image
import requests
import matplotlib.pyplot as plt

# %config InlineBackend.figure_format = 'retina'

# import torch
# from torch import nn
# from torchvision.models import resnet50
# import torchvision.transforms as T
from custom_parsing import custom_parsing_class
import numpy as np

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
import json
from pycocotools import mask as mask_util
from pathlib import Path


# torch.set_grad_enabled(False);





cps = custom_parsing_class()
# dtfc = detr_util_functions_class()


#path /home/keti/ap_ws/mpegdataset/test_utils/test_images-20250715T082534Z-1-001/test_images

# imgdir_path='/home/keti/ap_ws/mpegdataset/test_utils/test_images-20250715T082534Z-1-001/test_images/'

# impath=imgdir_path+'0012.jpg'# bin에서 읽어서 해야댐

binary_path = '/root/colmap/sparse/0/images.bin'
cambinpath='/root/colmap/sparse/0/cameras.bin'
points3D_bin_path='/root/colmap/sparse/0/points3D.bin'
# im = Image.open(impath)

# json_string = ""
# with open('annotations.json', 'w', encoding='utf-8') as f:
#     f.write(json_string)

# 1. `json.load()`를 사용하여 파일을 읽습니다.
with open('/home/keti/ap_ws/Grounded-SAM-2/outputs/grounded_sam2_local_demo/grounded_sam2_all_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# with open('grounded_sam2_all_results.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

# 2. 딕셔너리 구조를 이용해 원하는 데이터에 접근합니다.
print('data len:',len(data))


iter=0
matching_info={}

for idx in data:# length : number of images
    # iter+=1
    # if iter > 3:
    #     break
    # print(f"Image Path: {idx['image_path']}")
    img_n=Path(idx['image_path'])
    img_idname=img_n.stem
    img_id=int(img_idname)
    # print(img_id)
    matching_info.setdefault(img_id, {})
    for i, annotation in enumerate(idx['annotations']):
        cls_name= annotation['class_name']
        if cls_name == 'black hole':
            # print(f"Annotation {i+1}:")
            # print(f"  Class Name: {annotation['class_name']}")
            # print(f"  Bounding Box: {annotation['bbox']}")
            # print(f"  Score: {annotation['score']}")
            mask_info=annotation['segmentation']
            mask = mask_util.decode(mask_info)  # shape = (H, W, 1) or (H, W)
            mask = np.squeeze(mask)       # (H, W)로 변환
            matching_info[img_id][cls_name]=mask
       
print(matching_info.keys())

# app_points_np = np.stack(app_points)
points3D=cps.read_points3D_binary(points3D_bin_path)  # points3D는 Point3D 객체의 딕셔너리 형태로 반환됨
images_extrin=cps.read_images_binary(binary_path)
for n,p in enumerate(points3D):
    # anomaly_features=[]
    # if n >1:
        # break
    # print("p:",p)
    # print('p.xyz:',points3D[p].xyz)
    # print('p.rgb:',points3D[p].rgb)
    # print('p.id:',points3D[p].id)
    # print('p.images_ids:',points3D[p].image_ids)
    id_list=points3D[p].image_ids#image_id



    for ls_idx in id_list:#images_id list
        if ls_idx in matching_info:
            matching_class=list(matching_info[ls_idx].keys())
            for cls_n in matching_class:
                # if matching_info[ls_idx][cls_n].any():
                mask=matching_info[ls_idx][cls_n]

        # print('img.xys:',images_extrin[ls_idx].xys)
        # print('img.point3d_id:',images_extrin[ls_idx].point3D_ids)
                matching_dict = {value: index for index, value in enumerate(images_extrin[ls_idx].point3D_ids) if value != -1}
                # matching_dict = {value: index for index, value in enumerate(images_extrin[ls_idx].point3D_ids) if value == p} # point3d_id : index
        # matching_xysdict = {value: index for index, value in enumerate(images_extrin[ls_idx].xys) if value == p}
                # print(images_extrin[ls_idx].xys[matching_dict[points3D[p].id]])
                pixel=np.array(images_extrin[ls_idx].xys[matching_dict[points3D[p].id]])
                pixel=np.round(pixel).astype(int)  # xys는 float형태니깐 round로 반올림해야함
                # print('pixel:',pixel)
                if mask[pixel[1]][pixel[0]] == 1:  # mask는 2D numpy 배열
                    if points3D[p].anomaly_features is None:
                        # print('update anomaly_features')
                        # updated = points3D[p]._replace(anomaly_features=np.array([1]))
                        updated = points3D[p]._replace(anomaly_features=1)
                        points3D[p]=updated
                    # points3D[p]._replace(anomaly_features=1)
        # print('pixel:',pixel)
        # print(matching_dict)


print('hi)')
print(mask.shape)
print(points3D[5369].anomaly_features)
check_val = {value: index for index, value in enumerate(images_extrin[1].point3D_ids) if value != -1}
# print('check_val:',check_val)
point_ls=list(check_val.keys())
# updated=points3D[1914]._replace(anomaly_features=np.array([1]))
# points3D[1914]=updated
print()
print('iteration start')
print()
for x in point_ls:
    if points3D[x].anomaly_features is not None:
        print('x:',x,points3D[x].anomaly_features)
    # print(points3D[x].image_ids)
    # print(points3D[x].anomaly_features)
# print(points3D[1914].anomaly_features)
print('end')






# xys_ls=list(check_val.values())
# print('xys_ls:',xys_ls)
# for accs in xys_ls:
#     print('xys:',images_extrin[1].xys[accs])

# rounded_list = [[round(coord) for coord in images_extrin[1].xys[accs]] for accs in xys_ls]

# print(rounded_list)
# # import cv2
# import matplotlib.image as mpimg
# img_dir_path='/home/keti/ap_ws/mpegdataset/sample5506/0001.jpg'
# img = mpimg.imread(img_dir_path)
# red_color=(0,0,255)
# thickness=-1
# x_coords, y_coords = zip(*rounded_list)
# fig, ax = plt.subplots(1)

# y1_coords, x1_coords = np.where(matching_info[1]['black hole'] == 1)

# # Figure의 여백을 없애 이미지에 딱 맞게 설정
# fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

# # 4. 이미지 배경 표시
# ax.imshow(img)

# # 5. scatter 함수로 점들 그리기
# ax.scatter(x_coords, y_coords, c='red', s=5, marker='o')
# ax.scatter(x1_coords, y1_coords, c='blue', s=0.1, marker='o')
# # 6. 축 숨기기
# ax.axis('off')

# # 7. 화면에 최종 결과 표시
# # plt.show()
# plt.savefig('comparing.png', dpi=300, bbox_inches='tight')






#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# # 3D 좌표와 색상 추출
# xyz = []
# rgb = []
# for pt in points3D.values():
#     xyz.append(pt.xyz)
#     rgb.append(pt.rgb / 255.0)  # matplotlib은 0~1 범위의 색상 사용

# import numpy as np
# xyz = np.stack(xyz)
# rgb = np.stack(rgb)

# # app_points가 [[x, y, z], ...] 형태 또는 (M,3) np.array라면
# app_points_np = np.stack(app_points)  # 필요시

# # 3D 플롯팅
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
# # 1) COLMAP 포인트 (원래 점들, 색상별)
# ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=1, label="COLMAP Points")
# # 2) 새로 구한 app_points (빨간 점, 크게)
# ax.scatter(app_points_np[:, 0], app_points_np[:, 1], app_points_np[:, 2],
#            c='r', s=60, marker='o', label="Triangulated Points")
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('COLMAP 3D Point Cloud + Triangulated Points')
# ax.legend()
# # plt.show()
# plt.savefig('3d_point_cloud_0723.png', dpi=300, bbox_inches='tight')
