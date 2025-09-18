import os
from PIL import Image
import requests
import cv2
# import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'retina'

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
from custom_parsing import custom_parsing_class
import numpy as np

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
import json
from pycocotools import mask as mask_util
from pathlib import Path
import random
# torch.set_grad_enabled(False);

class DETRdemo(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)

        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)
        
        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h), 
                'pred_boxes': self.linear_bbox(h).sigmoid()}

    #@@@@@@@@@@@util function for output
    def box_cxcywh_to_xyxy(self,x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)
    
    def rescale_bboxes(self,out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b
    
    def detect(self,im, transform):
        # mean-std normalize the input image (batch-size: 1)
        img = transform(im).unsqueeze(0)

        # demo model only support by default images with aspect ratio between 0.5 and 2
        # if you want to use images with an aspect ratio outside this range
        # rescale your image so that the maximum size is at most 1333 for best results
        assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

        # propagate through the model
        outputs = self(img)

        # keep only predictions with 0.7+ confidence
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.7

        # convert boxes from [0; 1] to image scales
        bboxes_scaled = self.rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
        return probas[keep], bboxes_scaled
    

#config initialization
# detr = DETRdemo(num_classes=91)
# state_dict = torch.hub.load_state_dict_from_url(
# url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
# map_location='cpu', check_hash=True)
# detr.load_state_dict(state_dict)
# detr.eval();

# # COCO classes
# CLASSES = [
#     'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#     'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
#     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
#     'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
#     'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
#     'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
#     'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
#     'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
#     'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
#     'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
#     'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
#     'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
#     'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
#     'toothbrush'
# ]

# # colors for visualization
# COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
#           [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# # standard PyTorch mean-std input image normalization
# transform = T.Compose([
#     T.Resize(800),
#     T.ToTensor(),
#     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
#@@@@@@@@@@@@@@@@@@@@@2
# for output bounding box post-processing

class detr_util_functions_class:
    
    def box_cxcywh_to_xyxy(self,x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)
    
    def rescale_bboxes(self,out_bbox, size):
        img_w, img_h = size
        b = box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b
    
    def detect(self,im, model, transform):
        # mean-std normalize the input image (batch-size: 1)
        img = transform(im).unsqueeze(0)

        # demo model only support by default images with aspect ratio between 0.5 and 2
        # if you want to use images with an aspect ratio outside this range
        # rescale your image so that the maximum size is at most 1333 for best results
        assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

        # propagate through the model
        outputs = model(img)

        # keep only predictions with 0.7+ confidence
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.7

        # convert boxes from [0; 1] to image scales
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
        return probas[keep], bboxes_scaled



cps = custom_parsing_class()
# dtfc = detr_util_functions_class()


#path /home/keti/ap_ws/mpegdataset/test_utils/test_images-20250715T082534Z-1-001/test_images

imgdir_path='/home/keti/ap_ws/mpegdataset/test_utils/test_images-20250715T082534Z-1-001/test_images/'

impath=imgdir_path+'0012.jpg'# bin에서 읽어서 해야댐

binary_path = '/root/colmap/sparse/0/images.bin'
cambinpath='/root/colmap/sparse/0/cameras.bin'
points3D_bin_path='/root/colmap/sparse/0/points3D.bin'
im = Image.open(impath)
# scores, boxes = detr.detect(im, transform)



#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
def rgb_thrsh(rgb1,rgb2,tolerance_s : int):
    # tolerance = tolerance_s / 255.0  # RGB 값은 0~255 범위이므로, tolerance를 0~1 범위로 변환
    tolerance = tolerance_s 
    r1,g1,b1 =rgb1
    r2,g2,b2 =rgb2#소수점 정규화
    return (abs(r1 - r2) < tolerance and abs(g1 - g2) < tolerance and abs(b1 - b2) < tolerance)
    
def get_pixel_color_cv2(image_path: str, coords: list) -> tuple:
    # print('image_path:',image_path)
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError
        
        x, y = coords
        
        # 1. 배열 인덱싱은 [y, x] 순서임을 주의
        # bgr_color1 = img[y, x]
        bgr_color = img[y, x]
        # bgr_color = bgr_color1.astype(np.float32) / 255.0
        # 2. BGR을 RGB로 변환 (순서 뒤집기)
        rgb_color = [bgr_color[2], bgr_color[1], bgr_color[0]]
        
        return rgb_color
        
    except FileNotFoundError:
        print(f"can not find img: {image_path}")
        return None
    except IndexError:
        print(f"exceed pixel img size")
        return None



json_string = ""
with open('annotations.json', 'w', encoding='utf-8') as f:
    f.write(json_string)

# 1. `json.load()`를 사용하여 파일을 읽습니다.
with open('/home/keti/ap_ws/Grounded-SAM-2/outputs/grounded_sam2_local_demo/grounded_sam2_all_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 딕셔너리 구조를 이용해 원하는 데이터에 접근합니다.
print('data len:',len(data))
colmap_points=cps.read_points3D_binary(points3D_bin_path)  # points3D는 Point3D 객체의 딕셔너리 형태로 반환됨
images_extrin=cps.read_images_binary(binary_path)

iter=0
matching_info={}
for idx in data:# length : number of images
    img_n=Path(idx['image_path'])
    img_idname=img_n.stem
    img_id=int(img_idname)
    matching_info.setdefault(img_id, {})

    for i, annotation in enumerate(idx['annotations']):
        cls_name= annotation['class_name']
        if cls_name == 'black hole':
            mask_info=annotation['segmentation']
            mask = mask_util.decode(mask_info)  # shape = (H, W, 1) or (H, W)
            mask = np.squeeze(mask)       # (H, W)로 변환
            matching_info[img_id][cls_name]=mask
            matching_info[img_id]['img_name']=idx['image_path']
            # print('shape',mask.shape)
# print('matching_info:',matching_info.items())
# origin=colmap_points.deepcopy()
for n,p in enumerate(colmap_points):
    id_list=colmap_points[p].image_ids#image_id
    for ls_idx in id_list:#images_id list
        if ls_idx in matching_info:#matching info에 id 가 있다면
            # matching_class=list(matching_info[ls_idx].keys())
            matching_class = [key for key in matching_info[ls_idx] if key != 'img_name']
            for cls_n in matching_class:
                mask=matching_info[ls_idx][cls_n]
                matching_dict = {value: index for index, value in enumerate(images_extrin[ls_idx].point3D_ids) if value != -1}
                pixel=np.array(images_extrin[ls_idx].xys[matching_dict[colmap_points[p].id]])
                pixel=np.round(pixel).astype(int)  # xys는 float형태니깐 round로 반올림해야함

                if mask[pixel[1]][pixel[0]] == 1:  # mask는 2D numpy 배열
                    rgb1=colmap_points[p].rgb
                    rgb2=get_pixel_color_cv2(matching_info[ls_idx]['img_name'],pixel)
                    # print(rgb1,rgb2)
                    if rgb_thrsh(rgb1,rgb2, tolerance_s=30):
                        if colmap_points[p].anomaly_features == 0:
                            updated = colmap_points[p]._replace(anomaly_features=1)
                            # print(f"origin point.xyz:{colmap_points[p].xyz}")
                            colmap_points[p]=updated
                            # print(f"updated point.xyz:{colmap_points[p].xyz}")
    # if (colmap_points[p].xyz != origin[p].xyz).any():  # xyz가 변경되었는지 확인
    #     print(f"Point ID: {p}, {colmap_points[p].xyz} -> {origin[p].xyz},")
    


counter1=0
counter2=0
counter3=0
for p in colmap_points:
    if colmap_points[p].anomaly_features == 1:
        # print(f"Point ID: {colmap_points[p].id},Point xyz: {colmap_points[p].xyz} Point rgb: {colmap_points[p].rgb} Anomaly Features: {colmap_points[p].anomaly_features}")
        counter1+=1
        if sum(colmap_points[p].rgb) < 15:  # RGB 값이 너무 어두운 경우
            counter3+=1
            print(f"Point ID: {colmap_points[p].id},Point xyz: {colmap_points[p].xyz} Point rgb: {colmap_points[p].rgb} Anomaly Features: {colmap_points[p].anomaly_features}")
    elif sum(colmap_points[p].rgb) < 15:
        # print(f"Point ID: {colmap_points[p].id},Point xyz: {colmap_points[p].xyz} Point rgb: {colmap_points[p].rgb} Anomaly Features: {colmap_points[p].anomaly_features}")
        counter2+=1

# print('couter1:',counter1)
# print('couter2:',counter2)
# print('couter3:',counter3)
#65 64없음
# print('image_id:',images_extrin.keys())
# for _ in range(10):
#     k=random.randint(1, 67)
#     print('image_id:',k,images_extrin[k].name)
cps.create_ply_from_colmap(
    filename='sparse_pc.ply',
    output_dir=Path('/root'),
    pts=colmap_points)



        #xys는 float형태니깐 round로 반올림해야함
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
