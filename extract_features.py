import os
from PIL import Image
import requests
# import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'retina'

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
from custom_parsing import custom_parsing_class
import numpy as np
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
detr = DETRdemo(num_classes=91)
state_dict = torch.hub.load_state_dict_from_url(
url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
map_location='cpu', check_hash=True)
detr.load_state_dict(state_dict)
detr.eval();

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
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

imgdir_path='/home/keti/ap_ws/gaussian-splatting/mpegdataset/test_utils/test_images-20250715T082534Z-1-001/test_images/'

impath=imgdir_path+'0012.jpg'# bin에서 읽어서 해야댐

binary_path = '/root/colmap/sparse/0/images.bin'
cambinpath='/root/colmap/sparse/0/cameras.bin'
points3D_bin_path='/root/colmap/sparse/0/points3D.bin'
im = Image.open(impath)
# scores, boxes = detr.detect(im, transform)


finding_class='toilet'
a_idx=CLASSES.index(finding_class)
import glob
jpg_files = glob.glob(f"{imgdir_path}*.jpg")

features_info=[]
for idx,img_n in enumerate(jpg_files):
    pil_img = Image.open(img_n)
    prob, boxes = detr.detect(pil_img, transform)
    for p,(xmin, ymin, xmax, ymax) in zip(prob,boxes):
        cl=p.argmax()
        if cl == a_idx:
            c_x=(xmin.item()+xmax.item())//2
            c_y=(ymin.item()+ymax.item())//2
            img_name=img_n.rsplit('/')[-1]
            # features_info.append([c_x,c_y,img_name])
            br,bg,bb=pil_img.getpixel((c_x,c_y))# check pixel value
            # features_info.append([c_x,c_y,img_name])
            features_info.append([c_x,c_y,img_name,br,bg,bb])
# print(scores)

print('featuresd_info:',features_info)
images_extrin=cps.read_images_binary(binary_path)
# for f1 in features_info:
#     matching_name = f1[-1]
#     print(matching_name)
#     images_extrin=cps.read_images_binary(binary_path)
#     matched_ids = [image_id for image_id, img in images_extrin.items() if img.name == matching_name]

# binary_matching_param=[]

# for f1 in features_info:
#     print(f1[-1])


# for image_id, img in images_extrin.items():
    # print('image_id:',image_id)
    # print('img.name:',img.name)

matched_ids=[]
matched_ids = [
    image_id
    for f1 in features_info
    for image_id, img in images_extrin.items()
    if img.name == 'frame_0'+f1[-4]#@@@@
]
camera_intrin=cps.read_cameras_binary(cambinpath)
# udst=cps.undistorting(camera_intrin,features_info)#need to be changed
udst,color_list=cps._undistorting(camera_intrin,features_info)#need to be changed
print('color:',color_list)
cwl=[]
dwl=[]
lscwdw=[]
app_points=[]
for idx,iter in enumerate(matched_ids):

    buf1=images_extrin[iter]
    # print('printqvec',buf1.qvec)
    R=cps.qvec2rotmat(buf1.qvec)
    # print('printtvec',buf1.tvec)
    # bufcw,bufdw=cps.cal_ray(R,buf1.tvec,udst) #@@@@
    lscwdw=cps._cal_ray(R,buf1.tvec,udst[0][idx]) #@@@@
    print('output',lscwdw)



    # cwl.append(bufcw)
    # dwl.append(bufdw)
    
    cwl.append(lscwdw[0][0])
    dwl.append(lscwdw[0][1])

print('cwl:',cwl)
print('cwl[0]:',cwl[0])
print('dwl:',dwl)
# pxyz=cps.triangulate_multi_rays(cwl,dwl)
pxyz,nbypts=cps.triangulate_multi_rays(cwl,dwl)
app_points.append(pxyz)
app_points.extend(nbypts)
#rgb넣어주기
print('app_points',app_points)






print('camera_intrin:',camera_intrin)

# print(boxes)
print(matched_ids)
print('ok')
print(len(camera_intrin))
print(type(camera_intrin))

cps.write_post_process_points(app_points,color_list)
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


# app_points_np = np.stack(app_points)
# points3D=cps.read_points3D_binary(points3D_bin_path)  # points3D는 Point3D 객체의 딕셔너리 형태로 반환됨

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
