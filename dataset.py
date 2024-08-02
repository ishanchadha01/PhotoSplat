import os
from typing import NamedTuple

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

from utils.misc import get_focal2fov, get_world2view, get_proj_matrix


class Camera(NamedTuple):
    """
    Image, pose, and other metadata for each training image
    """
    R: np.array
    t: np.array
    img: np.array
    depth_map: np.array
    img_path : str
    time : float
    mask : np.array
    znear : float
    zfar : float
    xfov : float
    yfov : float
    full_proj_transform: torch.Tensor
    camera_center: torch.Tensor


class GSDataset(Dataset):
    def __init__(self, images, poses_bounds, masks, depths, image_paths):
        self.images = images
        self.poses = poses_bounds[:, :12]
        self.h = poses_bounds[0, 12]
        self.w = poses_bounds[0, 13]
        self.f = poses_bounds[0, 14]
        self.bounds = poses_bounds[:, -2:]
        self.masks = masks
        self.depths = depths
        self.image_paths = images

        # Populate cameras
        self.cameras = []
        for idx in range(len(self.images)):
            # get img and pose
            img_path = self.image_paths[idx]
            img = self.images[idx]
            pose = self.poses[idx].reshape((3,4))
            c2w = np.concatenate((pose, np.array([[0, 0, 0, 1]])), axis=0) # 4x4
            # R = c2w[:3,:3] # c2w rotation
            # t = -R.T @ c2w[:3,3] # w2c translation
            # lets just use w2c fully
            R = c2w[:3,:3].T
            t = -R @ c2w[:3,3] # transpose of R has already been taken

            # get mask and depths
            mask = self.masks[idx]
            depths = self.depths[idx]

            # get other info
            xfov = get_focal2fov(self.f, self.w)
            yfov = get_focal2fov(self.f, self.h)
            znear = self.bounds[idx, 0]
            zfar = self.bounds[idx, 1]
            time = idx / len(self.images)

            ###TODO: need to check and incorporate this
            # self.world_view_transform = torch.tensor(get_world2view(R, t)).transpose(0, 1)
            world_view_transform = torch.zeros((4,4))
            world_view_transform[:3,:3] = torch.from_numpy(R)
            world_view_transform[:3, 3] = torch.from_numpy(t)
            proj_mat = get_proj_matrix(znear=znear, zfar=zfar, fovX=xfov, fovY=yfov).transpose(0,1) # transposing makes it row-wise
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(proj_mat.unsqueeze(0))).squeeze(0)
            # camera_center = world_view_transform.inverse()[3, :3]
            camera_center = torch.from_numpy(t)

            ###

            cam = Camera(R=R, t=t, img=img, depth_map=depths, img_path=img_path, time=time, 
                         mask=mask, znear=znear, zfar=zfar, xfov=xfov, yfov=yfov, 
                         full_proj_transform=full_proj_transform, camera_center=camera_center)
            self.cameras.append(cam)

        # set camera extent
        self.get_nerfplusplus_norm()

    def get_nerfplusplus_norm(self):
        """
        Get size of scene using Nerf++ method.
        """
        def get_center_and_diag(cam_centers):
            cam_centers = np.hstack(cam_centers)
            avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
            center = avg_cam_center
            dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
            diagonal = np.max(dist)
            return center.flatten(), diagonal

        cam_centers = []
        for cam in self.cameras:
            W2C = get_world2view(cam.R, cam.t)
            C2W = np.linalg.inv(W2C)
            cam_centers.append(C2W[:3, 3:4])

        center, diagonal = get_center_and_diag(cam_centers)
        radius = diagonal * 1.1 # want it slightly larger than min and max
        translate = -center

        self.camera_extent = {"translate": translate, "radius": radius}

    def __getitem__(self, idx):
        """
        Returns relevant camera
        """
        return self.cameras[idx]

    def __len__(self):
        return len(self.cameras)
