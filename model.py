import torch
import numpy as np
import pycolmap

from deformation import DeformNet
from utils.misc import PointCloud, inverse_sigmoid, get_expon_lr_func, distCUDA2, strip_symmetric, build_scaling_rotation, convert_quat2rot
from utils.sh_utils import convert_rgb2sh, eval_sh

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

import math


class PhotoSplatter():
    """
    Gaussian splatting model with call to deformation network.
    """

    def __init__(self, args):

        # photometric calibration
        self.y = args['gamma']
        self.g = args['gain']
        self.k = args['cosine_decay']

        # data structures initialization
        self.init_method = args['gaussian_init_method']
        self._means = torch.empty(0) # points [N,3]
        self._rotations = torch.empty(0) # rot for each point [N,3]
        self._scalings = torch.empty(0) # scale for each point [N,3]
        self._opacities = torch.empty(0) # alphas for each point [N,3]
        self._feats_color = torch.empty(0) # [N,3,s1]
        self._feats_rest = torch.empty(0) # [N,3,sh_slots-1]
        self._max_radii2d = torch.empty(0) # [N] max radius for each gaussian once splatted
        self.means_gradient_accum = torch.empty(0) # [N] gradient accumulation for means 
        # to check if grad is high enough for densification
        self.denom = torch.empty(0) # [N] keeps track of number of points considered during densification
        self.sh_degree = args['sh_degree']
        self.initialize_gaussians()

        # build covariance matrix dynamically when needed
        def build_covariance_from_scaling_rotation(scaling, rotation): #TODO leaving out scaling modifier for now
            # taking in scale and quaternion and create scaled rotation of size Nx6 since only upper triangle is needed
            L = build_scaling_rotation(scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        self.covariance = build_covariance_from_scaling_rotation

        # activation functions
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        # optimization params
        self.opacity_thresh_low = args['opacity_thresh_low'] # for pruning
        self.size_thresh = args['size_thresh'] # if scaling of gaussian is greater than this, then split
        self.percent_dense = args['percent_dense'] # maximum proportion of scene occupied by gaussians
        # otherwise clone, for densification
        self.density_grad_thresh = args['densify_grad_thresh']
        self.optimize_until_iter = args['optimize_until_iter']

        # delta net and deformation params
        self._deform_mask = torch.empty(0) # [N] binary mask for points which have been deformed
        self.deform_net = DeformNet()
        self._deform_accum = torch.empty(0) # [N,3], how much each point has been cumulatively deformed over all iters

        # training params
        self.position_lr_init = args['position_lr_init']
        self.position_lr_final = args['position_lr_final']
        self.position_lr_delay_mult = args['position_lr_delay_mult']
        self.position_lr_max_steps = args['position_lr_max_steps']

        self.deformation_lr_init = args['deformation_lr_init']
        self.deformation_lr_final = args['deformation_lr_final']
        self.deformation_lr_delay_mult = args['deformation_lr_delay_mult']

        self.grid_lr_init = args['grid_lr_init']
        self.grid_lr_final = args['grid_lr_final']

        self.feature_lr = args['feature_lr']
        self.opacity_lr = args['opacity_lr']
        self.scaling_lr = args['scaling_lr']
        self.rotation_lr = args['rotation_lr']

        self.spatial_lr_scale = args['spatial_lr_scale']


    def initialize_gaussians(self, img=None):
        ## TODO: set all colors to red for now?
        if self.init_method == 'colmap':
            pts = None
            # TODO: use pycolmap to extract features, perform exhaustive matching, and then get sparse pts as [N,3]
        elif self.init_method == 'photometric':
            if img is None:
                raise Exception("Must pass in first image if using photometric Gaussian initialization")
            pts = self._init_pts_photometric(img, self.k, self.y, self.g)
            # TODO: init gaussians based off of photometric approach
        elif self.init_method == 'random':
            pts = self._init_pts_random() # for testing purposes probably
        else:
            raise Exception("No Gaussian initialization method provided in config file")

        colors = torch.zeros((*pts.shape, 3)) + torch.tensor([255,0,0])
        normals = torch.tensor([0,0,1]).repeat(*pts.shape, 1)

        pt_cloud = PointCloud(pts=pts, colors=colors, normals=normals)

        self._init_from_pt_cloud(pt_cloud)

    
    def _init_pts_photometric(self, img):
        #TODO
        pass

    
    def _init_pts_random(self):
        # initialize 1000 pts in 0 to 1 scale
        return np.random.rand(1000,3)

    
    def _init_from_pt_cloud(self, pt_cloud : PointCloud):
        fused_point_cloud = pt_cloud.pts.float().cuda()
        fused_color = convert_rgb2sh(pt_cloud.colors.float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.sh_degree + 1) ** 2)).float().cuda() # [N,3,sh_slots]
        features[:, :3, 0 ] = fused_color # only color used for first sh slot, before using rgb2sh, [N,3,1]
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialization : ", fused_point_cloud.shape[0])

         # put pts in kmeans data structure, mean of distance to closest 3 pts for each  pt
        dist2 = torch.clamp_min(distCUDA2(pt_cloud.pts.float().cuda()), 0.0000001)

        # gets sqrt for scale, isotropic covariance
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)

        # all gaussians oriented have no rotation
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device='cuda')
        rots[:, 0] = 1

        # scale opacities into valid range
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # data structures init
        num_pts = self._means.shape[0]
        self._means = torch.nn.Parameter(fused_point_cloud.requires_grad_(True))
        self.deform_net = self._deltas.to('cuda') 
        self._feats_color = torch.nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._feats_rest = torch.nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scalings = torch.nn.Parameter(scales.requires_grad_(True))
        self._rotations = torch.nn.Parameter(rots.requires_grad_(True))
        self._opacities = torch.nn.Parameter(opacities.requires_grad_(True))
        self._max_radii2D = torch.zeros((num_pts), device='cuda') #TODO: update to be num gaussians? max 2d radii for splatted nax radius
        self._deform_mask = torch.gt(torch.ones((num_pts),device='cuda'),0)

        # training setup
        self.means_gradient_accum = torch.zeros((num_pts, 1), device='cuda')
        self.denom = torch.zeros((num_pts, 1), device='cuda')
        self._deform_accum = torch.zeros((num_pts,3),device='cuda')

        training_params = [
            {'params': [self._means], 'lr': self.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': list(self.deform_net.get_mlp_parameters()), 'lr': self.deformation_lr_init * self.spatial_lr_scale, "name": "deformation"},
            {'params': list(self.deform_net.get_grid_parameters()), 'lr': self.grid_lr_init * self.spatial_lr_scale, "name": "grid"},
            {'params': [self._feats_color], 'lr': self.feature_lr, "name": "f_color"},
            {'params': [self._feats_rest], 'lr': self.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacities], 'lr': self.opacity_lr, "name": "opacity"},
            {'params': [self._scalings], 'lr': self.scaling_lr, "name": "scaling"},
            {'params': [self._rotations], 'lr': self.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(training_params, lr=0.0, eps=1e-15)
        
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=self.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=self.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=self.position_lr_delay_mult,
                                                    max_steps=self.position_lr_max_steps)
        self.deformation_scheduler_args = get_expon_lr_func(lr_init=self.deformation_lr_init*self.spatial_lr_scale,
                                                    lr_final=self.deformation_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=self.deformation_lr_delay_mult,
                                                    max_steps=self.position_lr_max_steps) 
        self.grid_scheduler_args = get_expon_lr_func(lr_init=self.grid_lr_init*self.spatial_lr_scale,
                                                    lr_final=self.grid_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=self.deformation_lr_delay_mult,
                                                    max_steps=self.position_lr_max_steps)   


    def remove_gaussian(self, idx):
        '''
        Remove Gaussian at idx and fill hole in list with end value.
        Method used for debugging, in practice pruned points are masked out.
        '''
        self.denom -= 1
        self._means[idx] = self.means.pop()
        self._rotations[idx] = self.rotations.pop()
        self._scalings[idx] = self.scalings.pop()
        self._opacities[idx] = self.opacities.pop()


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if "xyz" in param_group["name"]:
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            if  "grid" in param_group["name"]:
                lr = self.grid_scheduler_args(iteration)
                param_group['lr'] = lr
            if "deformation" in param_group["name"]:
                lr = self.deformation_scheduler_args(iteration)
                param_group['lr'] = lr


    def render(self, camera, bg_color, is_fine, compute_cov3D_python, convert_SHs_python, debug):
    
        # Set up rasterization configuration
        tanfovx = math.tan(camera.xfov * 0.5)
        tanfovy = math.tan(camera.yfov * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(camera.image.shape[0]),
            image_width=int(camera.image.shape[1]),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform.cuda(),
            projmatrix=camera.full_proj_transform.cuda(),
            sh_degree=self.means.active_sh_degree,
            campos=camera.camera_center.cuda(),
            prefiltered=False,
            debug=debug
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        # create local vars
        screenspace_points = torch.zeros_like(self._means, dtype=self._means.dtype, requires_grad=True, device="cuda")
        screenspace_points.retain_grad()
        means_3d = self._means
        means_2d = screenspace_points
        opacities = self._opacities

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3d_precomp = None

        if compute_cov3D_python:
            cov3D_precomp = self.covariance(self._scalings, self._rotations)
        else:
            scales = self._scalings
            rotations = self.rotations
        deformed_pts_mask = self._deform_mask

        # create empty time array
        time = torch.tensor(camera.time).to(means_3d.device).repeat(means_3d.shape[0],1)

        # only deform during fine iterations
        if not is_fine:
            means3D_deform, scales_deform, rotations_deform, opacity_deform = means_3d, scales, rotations, opacities
        else:
            means3D_deform, scales_deform, rotations_deform, opacity_deform = self._deform_net(means_3d[deformed_pts_mask], scales[deformed_pts_mask], 
                                                                            rotations[deformed_pts_mask], opacities[deformed_pts_mask],
                                                                            time[deformed_pts_mask])

        # dont perform gradient when accumulating deformation in deform accum tracker for each gaussian
        with torch.no_grad():
            self._deform_accum[deformed_pts_mask] += torch.abs(means3D_deform - means_3d[deformed_pts_mask])

        means3d_final = torch.zeros_like(means_3d)
        rotations_final = torch.zeros_like(rotations)
        scales_final = torch.zeros_like(scales)
        opacities_final = torch.zeros_like(opacities)

        # set deformed pts
        means3d_final[deformed_pts_mask] =  means3D_deform
        rotations_final[deformed_pts_mask] =  rotations_deform
        scales_final[deformed_pts_mask] =  scales_deform
        opacities_final[deformed_pts_mask] = opacity_deform

        # set undeformed pts
        means3d_final[~deformed_pts_mask] = means_3d[~deformed_pts_mask]
        rotations_final[~deformed_pts_mask] = rotations[~deformed_pts_mask]
        scales_final[~deformed_pts_mask] = scales[~deformed_pts_mask]
        opacities_final[~deformed_pts_mask] = opacities[~deformed_pts_mask]

        # perform activations on remaining fields
        scales_final = self.scaling_activation(scales_final)
        rotations_final =self.rotation_activation(rotations_final)
        opacities = self.opacity_activation(opacities)


        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        # TODO assuming that we never override color
        if convert_SHs_python:
            shs_view = self._feats_color.transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2) #TODO why is this transposing? could be a source of error
            dir_pp = (self._means - camera.camera_center.cuda().repeat(self._feats_color.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized) #TODO need to write this in utils
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = self._feats_color

        # render and return
        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii, depth = rasterizer(
            means3D = means3d_final,
            means2D = means_2d,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacities,
            scales = scales_final,
            rotations = rotations_final,
            cov3D_precomp = cov3D_precomp
        )

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image,
                "depth": depth,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0, # cull gaussians
                "radii": radii,}


    def optimize(self, loss, psnr, iter, pbar, num_iters, timer, args, visibility_filter, radii, is_fine, camera_extent):

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr + 0.6 * ema_psnr_for_log
            num_pts = self._means.shape[0]
            if iter % 10 == 0:
                pbar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr:.{2}f}",
                                          "point":f"{num_pts}"})
                pbar.update(10)
            if iter == num_iters:
                pbar.close()

            # Log and save
            timer.pause()
            #TODO: logging
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], stage)
            # if (iter in saving_iterations):
            #     print("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(iteration, stage)
            # if dataset.render_process:
            #     if (iteration < 1000 and iteration % 10 == 1) \
            #         or (iteration < 3000 and iteration % 50 == 1) \
            #             or (iteration < 10000 and iteration %  100 == 1) \
            #                 or (iteration < 60000 and iteration % 100 ==1):
            #         render_training_image(scene, gaussians, video_cams, render, pipe, background, stage, iteration-1,timer.get_elapsed_time())
            timer.start()
            
            ## Densification and pruning

            # Keep track of max radii in image-space for pruning
            self._max_radii2D[visibility_filter] = torch.max(self._max_radii2D[visibility_filter], radii[visibility_filter]) # find the max between the 2 tensors
            # TODO gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

            #TODO: make sure that these params are named right
            # Get thresholds for pruning and densification
            opacity_thresh = self.opacity_thresh_low
            densify_thresh = self.density_grad_thresh

            # Densify
            if iter > self.args.densify_from_iter and iter % self.args.densify_interval == 0 and iter < self.densify_until_iter:
                size_threshold = 20 if iter > self.args.opacity_reset_interval else None
                self.densify(densify_thresh, camera_extent) #TODO
                
            # Prune
            if iter > self.args.prune_from_iter and iter % self.args.pruning_interval == 0 and iter < self.prune_until_iter:
                size_threshold = 40 if iter > self.args.opacity_reset_interval else None
                prune_mask = (self.get_opacity < opacity_thresh).squeeze()
                if size_threshold:
                    big_points_vs = self.max_radii2D > size_threshold
                    big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * camera_extent
                    prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
                self.prune_points(prune_mask)
                torch.cuda.empty_cache()
                
            # Reset opacity
            if iter % self.args.opacity_reset_interval == 0:
                print("reset opacity")
                self.reset_opacity() #TODO
                    
            # Optimizer step
            if iter < num_iters:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none = True)

            # if (iter in checkpoint_iterations): TODO logging
            #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
            #     torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    
    def densify(self, max_grad, extent):
        grads = self.means_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        ## Densify by cloning
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= max_grad, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense * extent)
        
        new_means = self.means[selected_pts_mask] 
        new_feats_color = self._feats_color[selected_pts_mask]
        new_feats_rest = self._feats_rest[selected_pts_mask]
        new_opacities = self._opacities[selected_pts_mask]
        new_scalings = self._scalings[selected_pts_mask]
        new_rotations = self._rotations[selected_pts_mask]
        new_deform_mask = self._deform_mask[selected_pts_mask]
        self.densification_postfix(new_means, new_feats_color, new_feats_rest, new_opacities, new_scalings, new_rotations, new_deform_mask)

        ## Densify by splitting
        # Extract points that satisfy the gradient condition
        n_init_points = self._means.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= max_grad, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*extent)
        if not selected_pts_mask.any():
            return
        
        # Double size of all arrays in order to make space for new gaussians created by splitting
        stds = self.get_scaling[selected_pts_mask].repeat(2, 1)
        averages = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=averages, std=stds)
        rots = convert_quat2rot(self._rotation[selected_pts_mask]).repeat(2, 1, 1)
        new_means = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scalings = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(2,1) / (0.8*2))
        new_rotations = self._rotation[selected_pts_mask].repeat(2,1)
        new_feats_color = self._features_dc[selected_pts_mask].repeat(2,1,1)
        new_feats_rest = self._features_rest[selected_pts_mask].repeat(2,1,1)
        new_opacities = self._opacity[selected_pts_mask].repeat(2,1)
        new_deform_mask = self._deformation_table[selected_pts_mask].repeat(2)
        self.densification_postfix(new_means, new_feats_color, new_feats_rest, new_opacities, new_scalings, new_rotations, new_deform_mask)

        # Prune points that exceeded the densification gradient
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(2 * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)


    def densification_postfix(self, new_means, new_feats_color, new_feats_rest, new_opacities, new_scalings, new_rotations, new_deform_mask):
        """
        Update param groups for optimizer.
        """
        tensors_dict = {
            "xyz": new_means,
            "f_color": new_feats_color,
            "f_rest": new_feats_color,
            "opacity": new_opacities,
            "scaling" : new_scalings,
            "rotation" : new_rotations,
            "deformation": new_deform_mask #TODO: why was this line commented out in original code
        }
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # params of optimizer group is list of length 1
            if len(group["params"])>1:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        self._means = optimizable_tensors["xyz"]
        self._feats_color = optimizable_tensors["f_dc"]
        self._feats_rest = optimizable_tensors["f_rest"]
        self._opacities = optimizable_tensors["opacity"]
        self._scalings = optimizable_tensors["scaling"]
        self._rotations = optimizable_tensors["rotation"]
        
        self._deform_mask = torch.cat([self._deform_mask, new_deform_mask],-1)
        self.means_gradient_accum = torch.zeros((self._means.shape[0], 1), device="cuda")
        self._deform_accum = torch.zeros((self._means.shape[0], 3), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")


    def prune_points(self, mask):
        """
        Prune points according to mask
        """
        valid_points_mask = ~mask
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) > 1:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][valid_points_mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][valid_points_mask]
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter((group["params"][0][valid_points_mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = torch.nn.Parameter(group["params"][0][valid_points_mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        self._means = optimizable_tensors["xyz"]
        self._feats_color = optimizable_tensors["f_dc"]
        self._feats_rest = optimizable_tensors["f_rest"]
        self._opacities = optimizable_tensors["opacity"]
        self._scalings = optimizable_tensors["scaling"]
        self._rotations = optimizable_tensors["rotation"]
        self._deform_accum = self._deform_accum[valid_points_mask]
        self.means_gradient_accum = self.means_gradient_accum[valid_points_mask]
        self._deform_mask = self._deform_mask[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]