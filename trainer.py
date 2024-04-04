from tqdm import tqdm
import os
from typing import NamedTuple

import torch
import numpy as np
import cv2

from model import PhotoSplatter
from dataset import GSDataset
from utils import Timer


class Trainer:
    def __init__(self, args):
        # extract relevant args
        self.coarse_iters = args['coarse_iters']
        self.fine_iters = args['fine_iters']
        image_height = args['image_height']
        image_width = args['image_width']
        images_dir = os.path.join(args['data_dir'], 'images')
        poses_bounds = np.load(os.path.join(args['data_dir'], 'poses_bounds.npy'))
        image_paths = [os.path.join(images_dir, fp) for fp in sorted(os.listdir(images_dir))]

        # Load images
        print("Loading images...")
        images = np.zeros((len(image_paths), image_height, image_width))
        for idx, img_path in tqdm(enumerate(image_paths)):
            images[idx] = cv2.imread(img_path)

        # Get masks
        masks_dir = os.path.join(args['data_dir'], 'masks')
        mask_paths = [os.path.join(masks_dir, fp) for fp in sorted(os.listdir(masks_dir))]
        print("Loading masks...")
        masks = np.zeros((len(mask_paths), image_height, image_width))
        for idx, mask_path in tqdm(enumerate(mask_paths)):
            masks[idx] = cv2.imread(mask_path)

        # Get depths
        depths_dir = os.path.join(args['data_dir'], 'depths')
        depth_paths = [os.path.join(depths_dir, fp) for fp in sorted(os.listdir(depths_dir))]
        print("Loading depths...")
        depths = np.zeros((len(depth_paths), image_height, image_width))
        for idx, depth_path in tqdm(enumerate(depth_paths)):
            depths[idx] = cv2.imread(depth_path)

        # Create relevant data structures
        self.dataset = GSDataset(images, poses_bounds, masks, depths, image_paths)
        pt_cloud = self.dataset.create_pt_cloud()
        self.model = PhotoSplatter(args, pt_cloud)
        self.timer = Timer()
        

    def train(self):
        self._train(self.coarse_iters)
        self._train(self.fine_iters)


    def _train(self, num_iters):
        """
        Main training loop
        """

        # training bookkeeping
        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)
        self.timer.start()

        # train model
        
        for iter in tqdm(range(num_iters)):
            pass
            # run iter of model
            # differentiable rasterization
            # color and depth loss
        
            self.timer.pause()
            # log
            self.timer.start()