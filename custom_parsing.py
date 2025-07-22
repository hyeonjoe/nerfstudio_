import numpy as np
import struct
from collections import namedtuple
import cv2

class custom_parsing_class:
    CameraModel = namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
    Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])
    BaseImage = namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
    
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



    def cal_ray(self,rot,tvec,info2):
        '''
        info1 : transform matrix
        info2 : undistorted [d_c,uv, K, dist]
        '''
        c_w= -rot.T @ np.array(tvec,dtype=float)
        d_w = rot.T @ info2[0]
        d_w /= np.linalg.norm(d_w)
        return [c_w,d_w] # ray : c_w + s*d_w

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
        for c, d in zip(cwl, dwl):
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
        return P
        

