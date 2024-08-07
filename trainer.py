from tqdm import tqdm
import os
from typing import NamedTuple
from random import randint

import torch
import numpy as np
import cv2
import lpips
from torchmetrics.regression import PearsonCorrCoef
import gc

from model import PhotoSplatter
from dataset import GSDataset
from utils.misc import Timer, print_gpu_memory_usage
from utils.loss_utils import compute_l1_loss, compute_lpips_loss, compute_ssim, compute_TV_loss, compute_psnr



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

        # extract args for render func
        self.compute_cov3D_python = args["compute_cov3D_python"]
        self.convert_SHs_python = args["convert_SHs_python"]
        self.debug = args["debug"]

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
        
        # Other hyperparams
        self.lambda_dssim = args['lambda_dssim']
        self.time_smoothness_weight = args["time_smoothness_weight"]
        self.lambda_lpips = args["lambda_lpips"]

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
        bg_color = torch.tensor([1,1,1], dtype=torch.float32, device="cuda")
        print(f"gpu usage {print_gpu_memory_usage()}")
        for iter in tqdm(range(num_iters)):
            iter_start.record()
            self.model.update_learning_rate(iter)
            if is_fine:
                start_idx = np.random() * (len(self.dataset) - 1) # start at random point in sequence if fine iters
            else:
                start_idx = 0

            images = []
            depths = []
            gt_images = []
            gt_depths = []
            masks = []
            radii_list = []
            visibility_filter_list = []
            viewspace_point_tensor_list = []

            camera_idx = randint(0, len(self.dataset)-1)
            iter_cams = [camera_idx] #TODO: this is how its done in endogaussian, rly only one cam per iter
            # for camera_idx in range(start_idx, len(self.dataset)):

            for camera_idx in iter_cams:
                camera = self.dataset[camera_idx]
                render_dict = self.model.render(
                    camera,
                    bg_color,
                    is_fine, 
                    self.compute_cov3D_python, 
                    self.convert_SHs_python, 
                    self.debug
                )
                rendered_image = render_dict["render"]
                rendered_depth = render_dict["depth"]
                viewspace_point_tensor = render_dict["viewspace_points"]
                visibility_filter = render_dict["visibility_filter"]
                radii = render_dict["radii"]
                gt_image = torch.from_numpy(camera.img).permute(2,0,1).cuda().float()
                gt_depth = torch.from_numpy(camera.depth_map).cuda().float()
                mask = torch.from_numpy(camera.mask).cuda()

                images.append(rendered_image.unsqueeze(0))
                depths.append(rendered_depth.unsqueeze(0))
                gt_images.append(gt_image.unsqueeze(0))
                gt_depths.append(gt_depth.unsqueeze(0))
                masks.append(mask.unsqueeze(0))
                viewspace_point_tensor_list.append(viewspace_point_tensor)
                radii_list.append(radii)
                visibility_filter_list.append(visibility_filter)

                #TODO: maybe add debug thing here, or get ri
                # del gt_image, gt_depth, mask
                # torch.cuda.empty_cache()
                # gc.collect()
                print(f"gpu usage on camera {camera_idx} {print_gpu_memory_usage()}")
            
            radii = torch.cat(radii_list,0).max(dim=0).values
            visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
            rendered_images = torch.cat(images,0)
            rendered_depths = torch.cat(depths, 0)
            gt_images = torch.cat(gt_images,0)
            gt_depths = torch.cat(gt_depths, 0)
            masks = torch.cat(masks, 0)

            # Loss computation
            l1_loss = compute_l1_loss(rendered_images, gt_images, masks) #TODO
            if (gt_depths!=0).sum() < 10:
                print("No depth loss")
                depth_loss = torch.tensor(0.).cuda()
            else:
                #TODO this has previously caused problems
                # rendered_depths_reshape = rendered_depths.reshape(-1, 1)
                # gt_depths_reshape = gt_depths.reshape(-1, 1)
                # mask_tmp = mask.reshape(-1)
                # rendered_depths_reshape = rendered_depths_reshape[mask_tmp!=0, :]
                # gt_depths_reshape = gt_depths_reshape[mask_tmp!=0, :]
                # depth_loss =  0.001 * (1 - PearsonCorrCoef(gt_depths_reshape, rendered_depths_reshape)) #TODO
                depth_loss = compute_l1_loss(rendered_depths, gt_depths)

            depth_tvloss = compute_TV_loss(rendered_depths)
            img_tvloss = compute_TV_loss(rendered_images)
            tv_loss = 0.03 * (img_tvloss + depth_tvloss)
            loss = l1_loss + depth_loss + tv_loss
            psnr = compute_psnr(rendered_images, gt_images, masks).mean().double()
            print(f"PSNR at iter {iter}: {psnr}")

            #TODO separate self args out in init of trainer
            if is_fine and self.time_smoothness_weight != 0: 
                tv_loss = self.model.compute_regulation(2e-2, 2e-2, 2e-2)
                loss += tv_loss
            if self.lambda_dssim != 0:
                ssim_loss = compute_ssim(rendered_images,gt_images) #TODO
                loss += self.lambda_ssim * (1.0-ssim_loss)
            if self.lambda_lpips !=0:
                lpips_model = lpips.LPIPS(net="vgg").cuda()
                lpipsloss = compute_lpips_loss(rendered_images, gt_images, lpips_model) #TODO lpips loss as well as lpips model
                loss += self.lambda_lpips * lpipsloss
            
            loss.backward()
            viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
            for idx in range(0, len(viewspace_point_tensor_list)):
                viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
            iter_end.record()
            
            # TODO: fill in args for this
            self.model.optimize(loss, psnr, iter, pbar, num_iters, timer, args, visibility_filter, radii, is_fine, camera_extent)
        
            self.timer.pause()
            # log
            self.timer.start()