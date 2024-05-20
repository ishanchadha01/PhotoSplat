import torch
import numpy as np
import pycolmap

from utils.misc import PointCloud, inverse_sigmoid, get_expon_lr_func, distCUDA2
from utils.sh_utils import convert_rgb2sh


class PhotoSplatter(torch.nn.Module):
    """
    """

    def __init__(self, args):
        super().__init__()
        
        # photometric calibration
        self.y = args.gamma
        self.g = args.gain
        self.k = args.cosine_decay

        # data structures initialization
        self.init_method = args.gaussian_init_method
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
        self.max_sh_degree = args.max_sh_degree
        self.initialize_gaussians()

        # optimization params
        self.opacity_thresh_low = args.opacity_thresh_low # for pruning
        self.percent_dense = args.percent_dense # if scaling of gaussian is greater than this, then split
        # otherwise clone, for densification

        # delta net and deformation params
        self._deform_mask = torch.empty(0) # [N] binary mask for points which have been deformed
        self.deform_net = DeformNet()
        self._deform_accum = torch.empty(0) # [N,3], how much each point has been cumulatively deformed over all iters

        # training params
        self.position_lr_init = args.position_lr_init
        self.position_lr_final = args.position_lr_final
        self.position_lr_delay_mult = args.position_lr_delay_mult
        self.position_lr_max_steps = args.position_lr_max_steps

        self.deformation_lr_init = args.deformation_lr_init
        self.deformation_lr_final = args.deformation_lr_final
        self.deformation_lr_delay_mult = args.deformation_lr_delay_mult

        self.grid_lr_init = args.grid_lr_init
        self.grid_lr_final = args.grid_lr_final

        self.feature_lr = args.feature_lr
        self.opacity_lr = args.opacity_lr
        self.scaling_lr = args.scaling_lr
        self.rotation_lr = args.rotation_lr

        self.spatial_lr_scale = args.spatial_lr_scale


    def initialize_gaussians(self, img=None):
        if self.init_method == 'colmap':
            pts = None
            # TODO: use pycolmap to extract features, perform exhaustive matching, and then get sparse pts as [N,3]
        elif self.init_method == 'photometric':
            if img is None:
                raise Exception("Must pass in first image if using photometric Gaussian initialization")
            pts = self._init_pts_photometric(img)
            # TODO: init gaussians based off of photometric approach
        elif self.init_method == 'random':
            pts = self._init_pts_random() # for testing purposes probably
        else:
            raise Exception("No Gaussian initialization method provided in config file")

        self.init_from_pt_cloud(pts)

    
    def _init_pts_photometric(self, img):
        #TODO
        pass

    
    def _init_pts_random(self):
        # initialize 1000 pts in 0 to 1 scale
        return np.random(1000,3)

    
    def _init_from_pt_cloud(self, pt_cloud : PointCloud):
        fused_point_cloud = pt_cloud.points.float().cuda()
        fused_color = convert_rgb2sh(pt_cloud.colors.float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda() # [N,3,sh_slots]
        features[:, :3, 0 ] = fused_color # only color used for first sh slot, before using rgb2sh, [N,3,1]
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialization : ", fused_point_cloud.shape[0])

         # put pts in kmeans data structure, mean of distance to closest 3 pts for each pt
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
        self._scaling = torch.nn.Parameter(scales.requires_grad_(True))
        self._rotation = torch.nn.Parameter(rots.requires_grad_(True))
        self._opacity = torch.nn.Parameter(opacities.requires_grad_(True))
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


    
    def forward(self, x):
        pass
        
    
    def optimize(self):
        for idx in range(self.num_gaussians):
            # perform pruning
            if self.opacities[idx] < self.opacity_thresh_coarse_low or self.check_size(idx):
                self.remove_gaussian(idx)
            
            # perform densification
                # check for over reconstruction

                # under reconstruction

    def remove_gaussian(self, idx):
        '''
        Remove Gaussian at idx and fill hole in list with end value
        '''
        self.num_gaussians -= 1
        self._means[idx] = self.means.pop()
        self._rotations[idx] = self.rotations.pop()
        self._scalings[idx] = self.scalings.pop()
        self._opacities[idx] = self.opacities.pop()


    

class DeformNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # create multires spatial delta feature planes
    
        # create single res temporal delta planes
    
        # create photometric delta planes