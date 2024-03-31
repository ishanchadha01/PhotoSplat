import os

from torch.utils.data import Dataset
import cv2


class GSDataset(Dataset):
    def __init__(self, images, poses_bounds):
        self.images = images
        self.poses = poses_bounds[:, :15]
        self.bounds = poses_bounds[:, -2:]


    def __getitem__(self, idx):
        """
        Returns image, pose, and near+far bounds at idx
        """
        img = self.images[idx]
        return img, self.poses[idx], self.bounds[idx]

    def __len__(self):
        return len(self.images)
    

class Camera:
    """
    Image, pose, and other metadata for each training image
    """

    def __init__(self):
        self.image = None
        self.R = None
        self.t = None
        self.near = None
        self.far = None