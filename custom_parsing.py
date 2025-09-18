import numpy as np
import struct
from collections import namedtuple
import cv2
from pathlib import Path
import torch
import json
class custom_parsing_class:
    CameraModel = namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
    Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])
    BaseImage = namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
    Point3D = namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs","anomaly_features"])


    CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
    }
    CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])
    CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS])
    def __init__(self):
        # self.txt_path = '/home/keti/ap_ws/gaussian-splatting/mpegdataset/test_utils/post_process_points.txt'
        self.txt_path = '/home/keti/ap_ws/mpegdataset/test_utils/post_process_points.txt'
    
    
    class Image(BaseImage):
        def qvec2rotmat(self):
            return qvec2rotmat(self.qvec)

    def read_cameras_binary(self,path_to_model_file):
        """
        see: src/base/reconstruction.cc
            void Reconstruction::WriteCamerasBinary(const std::string& path)
            void Reconstruction::ReadCamerasBinary(const std::string& path)
        """
        cameras = {}
        with open(path_to_model_file, "rb") as fid:
            num_cameras = self.read_next_bytes(fid, 8, "Q")[0]
            for _ in range(num_cameras):
                camera_properties = self.read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
                camera_id = camera_properties[0]
                model_id = camera_properties[1]
                model_name = self.CAMERA_MODEL_IDS[camera_properties[1]].model_name
                width = camera_properties[2]
                height = camera_properties[3]
                num_params = self.CAMERA_MODEL_IDS[model_id].num_params
                params = self.read_next_bytes(fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params)
                cameras[camera_id] = self.Camera(
                    id=camera_id, model=model_name, width=width, height=height, params=np.array(params)
                )
            assert len(cameras) == num_cameras
        return cameras

    def read_images_binary(self, path_to_model_file):
        """
            see: src/base/reconstruction.cc
                void Reconstruction::ReadImagesBinary(const std::string& path)
                void Reconstruction::WriteImagesBinary(const std::string& path)
        """
        images = {}
        with open(path_to_model_file, "rb") as fid:
            num_reg_images = self.read_next_bytes(fid, 8, "Q")[0]
            for _ in range(num_reg_images):
                binary_image_properties = self.read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
                image_id = binary_image_properties[0]
                qvec = np.array(binary_image_properties[1:5])
                tvec = np.array(binary_image_properties[5:8])
                camera_id = binary_image_properties[8]
                image_name = b""
                current_char = self.read_next_bytes(fid, 1, "c")[0]
                while current_char != b"\x00":  # look for the ASCII 0 entry
                    image_name += current_char
                    current_char = self.read_next_bytes(fid, 1, "c")[0]
                image_name = image_name.decode("utf-8")
                num_points2D = self.read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
                x_y_id_s = self.read_next_bytes(fid, num_bytes=24 * num_points2D, format_char_sequence="ddq" * num_points2D)
                xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))])
                point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
                images[image_id] = self.Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
            return images

    def read_next_bytes(self, fid, num_bytes, format_char_sequence, endian_character="<"):
        """Read and unpack the next bytes from a binary file.
        :param fid:
        :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
        :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
        :param endian_character: Any of {@, =, <, >, !}
        :return: Tuple of read and unpacked values.
        """
        data = fid.read(num_bytes)
        return struct.unpack(endian_character + format_char_sequence, data)

    def undistorting(self,info1,features_info):
        '''
        undistorting points
        '''
        u,v=features_info[0][0],features_info[0][1]
        if len(info1) !=1:
            print('check camera numbers, it is not 1')
            return None
        else:

            intrin_params = info1[1].params

        fx, fy, cx, cy, k1, k2, p1, p2 = intrin_params[:8]
        K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=np.float32)
        dist = np.array([k1, k2, p1, p2, 0], dtype=np.float32)  # OPENCV model 5개까지
        # (u,v) → undistorted 정규화 좌표
        uv = np.array([[u, v]], dtype=np.float32)    # shape (1,2)
        undist = cv2.undistortPoints(uv[None, ...], K, dist, P=None)  # (1,1,2)
        x_d, y_d = undist[0,0]
        d_c = np.array([x_d, y_d, 1.0])
        d_c /= np.linalg.norm(d_c)
        return [d_c,uv, K, dist]


    def _undistorting(self, info1, features_info):
        '''
            undistorting multiple points
        '''
        if len(info1) != 1:
            print('check camera numbers, it is not 1')
            return None

        intrin_params = info1[1].params
        fx, fy, cx, cy, k1, k2, p1, p2 = intrin_params[:8]
        K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0,  0,  1]], dtype=np.float32)
        dist = np.array([k1, k2, p1, p2, 0], dtype=np.float32)  # OPENCV model 5개까지

        # 여러 포인트의 (u,v) 좌표를 추출
        uvs = np.array([[f[0], f[1]] for f in features_info], dtype=np.float32)  # shape (N,2)
        print('uvs:',uvs )
        #컬러추출
        colls=[]
        for color in features_info:
            ftcol1=color[3]
            ftcol2=color[4]
            ftcol3=color[5]
            colls.append(np.array([ftcol1,ftcol2,ftcol3], dtype=np.float32))  # shape (N,3)

        
        # OpenCV undistortPoints를 위해 shape (1,N,2)로 변형
        # uvs_ = uvs[None, ...]  # shape (1, N, 2)
        uvs_ = uvs[:,None,:]  # shape (1, N, 2) 이걸 왜해야함?
        print('uvs_:',uvs_)
        undist = cv2.undistortPoints(uvs_, K, dist, P=None)  # (1, N, 2)
        # undist = undist[0]  # (N, 2)
        undist = undist[:,0,:]  # (N, 2)
        print('undist:',undist)
        # 각 undistorted 좌표를 homogeneous로, 정규화까지
        d_cs = []
        for x_d, y_d in undist:
            d_c = np.array([x_d, y_d, 1.0])
            d_c /= np.linalg.norm(d_c)
            d_cs.append(d_c)

        #마지막에 평균값 넣기
        mean_val= np.mean(colls, axis=0)
        colls.append(mean_val)
        # col_scaled=
        # 필요하다면 원래 uv, K, dist 등도 함께 리턴
        return [d_cs, uvs, K, dist],colls


    def cal_ray(self,rot,tvec,info2):
        '''
        info1 : transform matrix
        info2 : undistorted [d_c,uv, K, dist]
        '''
        c_w= -rot.T @ np.array(tvec,dtype=float)
        d_w = rot.T @ info2[0]
        d_w /= np.linalg.norm(d_w)
        return [c_w,d_w] # ray : c_w + s*d_w

    def _cal_ray(self, rot, tvec, info2):
        output = []
        '''
            rot : rotation matrix (3x3)
            tvec : translation vector (3,)
            info2 : undistorted [d_cs, uvs, K, dist]
            d_cs: list or np.ndarray of shape (N,3)
        '''
        print('info2:',info2)
        c_w = -rot.T @ np.array(tvec, dtype=float)  # 카메라 중심 (world 좌표)
        c_w = c_w.reshape(1, 3)


        d_cs = np.array(info2)  # (N, 3)
        if d_cs.ndim == 1:
            d_cs = d_cs.reshape(1, 3)  # (1, 3)

            
        # d_cs = np.array(info2[0])  # (N, 3)
        print('d_cs:',d_cs,d_cs.shape)
        # 모든 d_c에 대해 d_w 계산 (N, 3)
        d_ws = (rot.T @ d_cs.T).T  # (N, 3)

        # 방향 벡터 정규화
        d_ws = d_ws / np.linalg.norm(d_ws, axis=1, keepdims=True)

        # 결과: (N, 3), (N, 3) - 각각의 ray: c_w + s*d_w[i]
        output.append([c_w, d_ws])
        print('c_w:', c_w, c_w.shape)
        print('d_ws:', d_ws, d_ws.shape)
        # return [c_w, d_ws]
        return output


    def qvec2rotmat(self,q):
        """(qw,qx,qy,qz) -> 3x3 회전행렬"""
        qw, qx, qy, qz = q / np.linalg.norm(q)
        return np.array([
            [1-2*(qy*qy+qz*qz),     2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1-2*(qx*qx+qz*qz),     2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1-2*(qx*qx+qy*qy)]
        ], dtype=float)

    def triangulate_multi_rays(self,cwl,dwl):
        A = []
        b = []
        for bc, bd in zip(cwl, dwl):
            c=bc[0]
            d=bd[0]
            print('c,d',c,d)
            d = d / np.linalg.norm(d)
            S = np.array([
                [0, -d[2], d[1]],
                [d[2], 0, -d[0]],
                [-d[1], d[0], 0]
            ])
            A.append(S)
            b.append(S @ c)
        A = np.concatenate(A, axis=0)
        b = np.concatenate(b, axis=0)
        P, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        #@@@@@@@@@@@@@@@@
        Qs = []
        for bc, bd in zip(cwl, dwl):
            c=bc[0]
            d=bd[0]
            d = d / np.linalg.norm(d)
            s = np.dot(d, P - c)
            Q = c + s * d   # 광선 위의 근사점에 가장 가까운 점
            Qs.append(Q)

        print('Qs:', Qs)
        return P,Qs
        

    def read_points3D_binary(self,path_to_model_file):
        """
        see: src/base/reconstruction.cc
            void Reconstruction::ReadPoints3DBinary(const std::string& path)
            void Reconstruction::WritePoints3DBinary(const std::string& path)
        """
        points3D = {}
        default_feat = np.uint8(0)
        with open(path_to_model_file, "rb") as fid:
            num_points = self.read_next_bytes(fid, 8, "Q")[0]
            for _ in range(num_points):
                binary_point_line_properties = self.read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
                point3D_id = binary_point_line_properties[0]
                xyz = np.array(binary_point_line_properties[1:4])
                rgb = np.array(binary_point_line_properties[4:7])
                error = np.array(binary_point_line_properties[7])
                track_length = self.read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
                track_elems = self.read_next_bytes(fid, num_bytes=8 * track_length, format_char_sequence="ii" * track_length)
                image_ids = np.array(tuple(map(int, track_elems[0::2])))
                point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
                points3D[point3D_id] = self.Point3D(
                    id=point3D_id, xyz=xyz, rgb=rgb, error=error, image_ids=image_ids, point2D_idxs=point2D_idxs,anomaly_features=0

                )
        return points3D

    def post_process_points3d(self):
        xyz_list = []
        with open(self.txt_path, 'r') as f:
            for line in f:
                elems = line.strip().split()
                if len(elems) < 3:
                    continue  # 잘못된 라인 스킵
                x, y, z = map(float, elems[:3])
                xyz_list.append([x, y, z])
        return np.array(xyz_list, dtype=np.float32)


    def post_process_points3d_rgb(self):
        rgb_list = []
        with open(self.txt_path, 'r') as f:
            for line in f:
                elems = line.strip().split()
                if len(elems) < 3:
                    continue  # 잘못된 라인 스킵
                r,g,b = map(float, elems[-3:])
                rgb_list.append([r,g,b])
        return np.array(rgb_list, dtype=np.float32)
        
    def create_ply_from_colmap(self,
        filename: str,  output_dir: Path , pts:Point3D) -> None:
       

        # Load point Positions
        points3D = torch.from_numpy(np.array([p.xyz for p in pts.values()], dtype=np.float32))



        # Load point colours
        points3D_rgb = torch.from_numpy(np.array([p.rgb for p in pts.values()], dtype=np.uint8))

        points3D_anomaly = torch.from_numpy(np.array([p.anomaly_features for p in pts.values()], dtype=np.uint8))


        # write ply
        with open(output_dir / filename, "w") as f:
            # Header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points3D)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uint8 red\n")
            f.write("property uint8 green\n")
            f.write("property uint8 blue\n")
            f.write("property uint8 feat\n")
            f.write("end_header\n")

            for coord, color, feat in zip(points3D, points3D_rgb,points3D_anomaly):
                x, y, z = coord
                r, g, b = color
                y_out = z
                z_out = -y
                # f.write(f"{x:8f} {y:8f} {z:8f} {r} {g} {b} {feat}\n")
                f.write(f"{x:8f} {y_out:8f} {z_out:8f} {r} {g} {b} {feat}\n")

