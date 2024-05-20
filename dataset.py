import os
from typing import NamedTuple

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

from utils.misc import get_focal2fov, get_world2view


class Camera(NamedTuple):
    """
    Image, pose, and other metadata for each training image
    """
    R: np.array
    t: np.array
    img: np.array
    depths: np.array
    img_path : str
    time : float
    mask : np.array
    znear : float
    zfar : float
    xfov : float
    yfov : float


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
            t = -R @ c2w[:3,:3] # transpose of R has already been taken

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
            self.world_view_transform = torch.tensor(get_world2view(R, t)).transpose(0, 1)
            self.projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=xfov, fovY=self.yfov).transpose(0,1)
            self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
            self.camera_center = self.world_view_transform.inverse()[3, :3]

            ###

            cam = Camera(R=R, t=t, img=img, depths=depths, img_path=img_path, time=time, 
                         mask=mask, znear=znear, zfar=zfar, xfov=xfov, yfov=yfov)
            self.cameras.append(cam)


    def __getitem__(self, idx):
        """
        Returns relevant camera
        """
        return self.cameras[idx]

    def __len__(self):
        return len(self.cameras)
