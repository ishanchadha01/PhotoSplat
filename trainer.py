from tqdm import tqdm
import os
from typing import NamedTuple

import torch
import numpy as np
import cv2

from model import PhotoSplatter
from dataset import GSDataset
from utils.misc import Timer


class Trainer:
    def __init__(self, args):
        # extract relevant args
        self.coarse_iters = args['training']['coarse_iters']
        self.fine_iters = args['training']['fine_iters']
        images_dir = os.path.join(args['data_dir'], 'images')
        poses_bounds = np.load(os.path.join(args['data_dir'], 'poses_bounds.npy'))
        image_paths = [os.path.join(images_dir, fp) for fp in sorted(os.listdir(images_dir))]
        example_img = cv2.imread(image_paths[0])
        image_height = example_img.shape[0]
        image_width = example_img.shape[1]

        # Load images
        print("Loading images...")
        images = np.zeros((len(image_paths), *(example_img.shape)))
        for idx, img_path in tqdm(enumerate(image_paths)):
            images[idx] = cv2.imread(img_path)

        # Get masks
        masks_dir = os.path.join(args['data_dir'], 'masks')
        mask_paths = [os.path.join(masks_dir, fp) for fp in sorted(os.listdir(masks_dir))]
        print("Loading masks...")
        masks = np.zeros((len(mask_paths), image_height, image_width))
        for idx, mask_path in tqdm(enumerate(mask_paths)):
            masks[idx] = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Get depths
        depths_dir = os.path.join(args['data_dir'], 'depth')
        depth_paths = [os.path.join(depths_dir, fp) for fp in sorted(os.listdir(depths_dir))]
        print("Loading depths...")
        depths = np.zeros((len(depth_paths), image_height, image_width))
        for idx, depth_path in tqdm(enumerate(depth_paths)):
            depths[idx] = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

        # Create relevant data structures
        self.dataset = GSDataset(images, poses_bounds, masks, depths, image_paths)
        self.model = PhotoSplatter(args)
        self.timer = Timer()
        

    def train(self):
        self._train(self.coarse_iters, is_fine=False)
        self._train(self.fine_iters, is_fine=True)


    def _train(self, num_iters, is_fine):
        """
        Main training loop
        """

        # training bookkeeping
        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)
        self.timer.start()

        # train model
        bg_color = torch.tensor([1,1,1], dtype=torch.float32, device="cuda") #TODO: can we do this without cuda
        for iter in tqdm(range(num_iters)):
            pass
            # TODO
            self.model.update_learning_rate(iter)
            if is_fine:
                start_idx = np.random() * (len(self.dataset) - 1) # start at random point in sequence if fine iters
            else:
                start_idx = 0

            for camera_idx in range(start_idx, len(self.dataset)):
                camera = self.dataset[camera_idx]
                render_dict = self.model.render(camera, bg_color, is_fine, args) #TODO: pass in specific args depending on whats required

            


            # for pose in poses traversed thus far (need to make sure poses are passed in with dataset)
                # call render using current gaussian estimate
            
            # update params like lr and sh degree

            # differentiable rasterization (TODO when does this happen? and when does deformation happen)
            # color and depth loss
            # perform densification and pruning
        
            self.timer.pause()
            # log
            self.timer.start()