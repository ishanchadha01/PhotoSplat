from tqdm import tqdm
import os

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
        image_paths = [os.path.join(images_dir, fp) for fp in os.listdir(images_dir)]

        # Load images
        print("Loading images...")
        images = np.zeros((len(image_paths), image_height, image_width))
        for idx, img_path in tqdm(enumerate(image_paths)):
            images[idx] = cv2.imread(img_path)

        # Create relevant data structures
        self.dataset = GSDataset(images, poses_bounds)
        self.model = PhotoSplatter(args)
        self.timer = Timer()
        
        # Get point clouds and timestamps for model data

        

    def train(self):
        self.timer.start()
        self._train(self.coarse_iters)
        self._train(self.fine_iters)


    def _train(self, num_iters):
        """
        Main training loop
        """

        # do some warmup iterations

        # train model
        for epoch_i in tqdm(range(num_iters)):
            pass
            # run iter of model
            # differentiable rasterization
            # color and depth loss
        
            self.timer.pause()
            # log
            self.timer.start()