import torch
import torch.nn.functional as F
# import tinycudann as tcnn

from typing import Sequence, Collection, Iterable, Optional
import itertools

from utils.misc import normalize_aabb, initialize_weights


class DeformNet(torch.nn.Module):
    """
    Deformation network using HexPlanes approach.
    """

    def __init__(self, args):
        super().__init__()

        # initialize the timenet
        time_in = 2*args["time_base_pos_enc"]
        time_width = args["timenet_width"]
        time_out = args["timenet_out"]
        self.timenet = torch.nn.Sequential(
            torch.nn.Linear(time_in, time_width), torch.nn.ReLU(),
            torch.nn.Linear(time_width, time_out))
        
        # initialize hex plane field
        aabb = torch.tensor([[args["bounds"], args["bounds"], args["bounds"]],
                             [-args["bounds"], -args["bounds"], -args["bounds"]]])
        self.aabb = torch.nn.Parameter(aabb, requires_grad=False)
        self.grid_config =  [{ #TODO put this in config
                                'grid_dimensions': 2,
                                'input_coordinate_dim': 4,
                                'output_coordinate_dim': 32,
                                'resolution': [64, 64, 64, 25]
                                }]
        self.multiscale_res_multipliers = args["multires"]
        self.concat_features = True

        self.grids = torch.nn.ModuleList()
        self.grids_feat_dim = 0
        for res in self.multiscale_res_multipliers:

            # initialize coordinate grid
            config = self.grid_config[0].copy()
            config["resolution"] = [
                r * res for r in config["resolution"][:3]
            ] + config["resolution"][3:] # multi-res only on spatial planes, not temporal ones
            gp = self.init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )

            # Concatenate over feature len for each scale
            if self.concat_features: # shape[1] is out-dim 
                self.grids_feat_dim += gp[-1].shape[1]
            else:
                self.grids_feat_dim = gp[-1].shape[1]
            self.grids.append(gp)

        # initialize the deformation net
        self.D = args["deform_depth"]
        self.W = args["deform_width"]
        self.input_ch = (4+3)+((4+3)*args["scale_rotation_pos_enc"])*2
        self.input_ch_time = args["timenet_out"]

        if not args["fully_fused"]:
            self.pos_deform, self.scales_deform, self.rotations_deform, self.opacity_deform = self.create_net()
        else:
            self.pos_deform, self.scales_deform, self.rotations_deform, self.opacity_deform = self.create_fully_fused_net()

        # initialize buffers        
        self.register_buffer('time_base_pos_enc', torch.FloatTensor([(2**i) for i in range(args["time_base_pos_enc"])]))
        self.register_buffer('pos_base_enc', torch.FloatTensor([(2**i) for i in range(args["scale_rotation_pos_enc"])]))
        self.register_buffer('rotation_scaling_pos_enc', torch.FloatTensor([(2**i) for i in range(args["pos_base_pos_enc"])]))
        self.register_buffer('opacity_pos_enc', torch.FloatTensor([(2**i) for i in range(args["opacity_pos_enc"])]))
        self.apply(initialize_weights)

    def forward(self, ray_pts_embedding, scale_embedding, rotation_embedding, opacity_embedding, time_embedding):
        # Compute all time input for all deformations
        grid_feats = self.forward_hexplane(ray_pts_embedding[:,:3], time_embedding[:,:1])
        hidden = self.feature_out(grid_feats)
 
        # Position deformation
        dx = self.pos_deform(hidden)
        means_3d = ray_pts_embedding[:, :3] + dx

        # Scale deformation
        ds = self.scales_deform(hidden)
        scales = scale_embedding[:,:3] + ds

        # Rotation deform
        dr = self.rotations_deform(hidden)
        rotations = rotation_embedding[:,:4] + dr

        # Opacity deform
        do = self.opacity_deform(hidden)
        opacities = opacity_embedding[:,:1] + do

        return means_3d, scales, rotations, opacities
    
    def forward_hexplane(self, pos_embedding, time_embedding):
        pts = normalize_aabb(pts, self.aabb)
        pts = torch.cat((pos_embedding, time_embedding), dim=-1)  # [n_rays, n_samples, 4]
        pts = pts.reshape(-1, pts.shape[-1])
        feats = self.interpolate_multires_features(
            pts, 
            multires_grids=self.grids,
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features, 
            num_levels=len(self.grids)
        )
        if len(feats) < 1:
            feats = torch.zeros((0,1)).to(feats.device)
        return feats

    def create_net(self):
        mlp_out_dim = 0
        self.feature_out = [torch.nn.Linear(mlp_out_dim + self.grids_feat_dim ,self.W)]
        for i in range(self.D-1):
            self.feature_out.append(torch.nn.ReLU())
            self.feature_out.append(torch.nn.Linear(self.W,self.W))
        self.feature_out = torch.nn.Sequential(*self.feature_out)
        
        return  (
            torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(self.W,self.W),
                torch.nn.ReLU(),
                torch.nn.Linear(self.W, 3)
            ),\
            torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(self.W,self.W),
                torch.nn.ReLU(),
                torch.nn.Linear(self.W, 3)
            ),\
            torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(self.W,self.W),
                torch.nn.ReLU(),
                torch.nn.Linear(self.W, 4)
            ), \
            torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(self.W,self.W),
                torch.nn.ReLU(),
                torch.nn.Linear(self.W, 1)
            )
        )
    
    def create_fully_fused_net(self):
        #TODO get rid of repeated code between this and self.create_net()
        mlp_out_dim = 0
        self.feature_out = [torch.nn.Linear(mlp_out_dim + self.grids_feat_dim ,self.W)]
        for i in range(self.D-1):
            self.feature_out.append(torch.nn.ReLU())
            self.feature_out.append(torch.nn.Linear(self.W,self.W))
        self.feature_out = torch.nn.Sequential(*self.feature_out)

        ff_mlp1 = tcnn.Network(n_input_dims=self.W, 
            n_output_dims=3, 
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.W,
                "n_hidden_layers": 2,
            })

        ff_mlp2 = tcnn.Network(n_input_dims=self.W, 
            n_output_dims=3, 
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.W,
                "n_hidden_layers": 2,
            })

        ff_mlp3 = tcnn.Network(n_input_dims=self.W, 
            n_output_dims=4, 
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.W,
                "n_hidden_layers": 2,
            })

        ff_mlp4 = tcnn.Network(n_input_dims=self.W, 
            n_output_dims=1, 
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.W,
                "n_hidden_layers": 2,
            })

        return ff_mlp1, ff_mlp2, ff_mlp3, ff_mlp4

    def init_grid_param(
        self,
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5
    ):
        assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
        assert grid_nd <= in_dim

        has_time_planes = in_dim == 4
        coord_combos = list(itertools.combinations(range(in_dim), grid_nd))
        grid_coefs = torch.nn.ParameterList()
        for coord_combo in coord_combos:
            new_grid_coef = torch.nn.Parameter(torch.empty(
                [1, out_dim] + [reso[cc] for cc in coord_combo[::-1]]
            ))
            if has_time_planes and 3 in coord_combo:  # Initialize time planes to 1
                torch.nn.init.ones_(new_grid_coef)
            else:
                torch.nn.init.uniform_(new_grid_coef, a=a, b=b)
            grid_coefs.append(new_grid_coef)

        return grid_coefs

    def interpolate_multires_features(
                            self,
                            pts: torch.Tensor,
                            multires_grids: Collection[Iterable[torch.nn.Module]],
                            grid_dimensions: int,
                            concat_features: bool,
                            num_levels: int,
                            ) -> torch.Tensor:
        coord_combos = list(itertools.combinations(
            range(pts.shape[-1]), grid_dimensions)
        )
        multi_scale_interp = [] if concat_features else 0.
        
        for grid in multires_grids[:num_levels]:
            interp_space = 1.
            for ci, coo_comb in enumerate(coord_combos):

                # interpolate in plane
                feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
                interp_out_plane = (
                    self.grid_sample_wrapper(grid[ci], pts[..., coo_comb]).view(-1, feature_dim)
                )

                # compute product over planes
                interp_space = interp_space * interp_out_plane

            # combine over scales
            if concat_features:
                multi_scale_interp.append(interp_space)
            else:
                multi_scale_interp = multi_scale_interp + interp_space

        if concat_features:
            multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
        return multi_scale_interp

    def grid_sample_wrapper(
            self, 
            curr_grid: torch.Tensor, 
            masked_pts: torch.Tensor, 
            align_corners: bool = True):
        
        # if no batch dimension present, need to add it
        grid_dim = masked_pts.shape[-1]
        if curr_grid.dim() == grid_dim + 1:
            curr_grid = curr_grid.unsqueeze(0)
        if masked_pts.dim() == 2:
            masked_pts = masked_pts.unsqueeze(0)

        if grid_dim == 2 or grid_dim == 3:
            grid_sampler = F.grid_sample
        else:
            raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                    f"implemented for 2 and 3D data.")

        # Expand shape of pts to [num_pts, 1, num_grids-1, *pts_dim]
        masked_pts = masked_pts.view([masked_pts.shape[0]] + [1] * (grid_dim - 1) + list(masked_pts.shape[1:]))
        B, feature_dim = curr_grid.shape[:2]
        n = masked_pts.shape[-2]

        # Call torch functional grid sampler with computed inputs
        interp = grid_sampler(
            curr_grid,  # [B, feature_dim, reso, ...]
            masked_pts,  # [B, 1, ..., n, grid_dim]
            align_corners=align_corners,
            mode='bilinear', 
            padding_mode='border')
        interp = interp.view(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
        interp = interp.squeeze()  # [B?, n, feature_dim?]
        return interp

    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    
    def get_grid_parameters(self):
        return list(self.grids.parameters()) 


class HashHexPlaneField(torch.nn.Module):
    def __init__(self, bounds, kplanes_config, multires):
        super().__init__()
        aabb = torch.tensor([[bounds,bounds,bounds],
                             [-bounds,-bounds,-bounds]])
        self.aabb = torch.nn.Parameter(aabb, requires_grad=False)
        
        self.grid_config =  [kplanes_config] 
        self.multiscale_res_multipliers = multires
        self.concat_features = True

        # Init planes
        self.grids = None
        self.build_encoding()
        self.feat_dim = 16 * 8 # num levels * num features per level, multiply along 6 grids, figure out way to make 8 into 64
        print(f"feature_dim: {self.feat_dim}")


    def build_encoding(self):
        self.grids = torch.nn.ModuleList()
        #TODO: if this doesnt work, then make only spatial grids multires
        for _ in range(6):
            config = dict(
                otype="HashGrid",
                n_levels=16, # original multires has 4 levels
                n_features_per_level=8, # this is 2 for same reason as below
                log2_hashmap_size=22, # not sure how big this should be, bigger prolly better tho
                base_resolution=2 ** 5, # original multires is 4 levels 64,64,64,100, so trying 32,64,128,256
                per_level_scale=2.0, # we want to go up by powers of 2 between resolutions for now
            )
            # we're passing in xt, yt, zt, xy, yz, xz, so 2 inputs per grid
            grid = tcnn.Encoding(2, config)
            self.grids.append(grid)

        # # for time dont do multires
        # for _ in range(3):
        #     config = dict(
        #         otype="HashGrid",
        #         n_levels=4, # original multires has 4 levels
        #         n_features_per_level=8, # this is 2 for same reason as below
        #         log2_hashmap_size=22, # not sure how big this should be, bigger prolly better tho
        #         base_resolution=2 ** 5, # original multires is 4 levels 64,64,64,100, so trying 32,64,128,256
        #         per_level_scale=2.0, # we want to go up by powers of 2 between resolutions for now
        #     )
        #     # we're passing in xt, yt, zt, xy, yz, xz, so 2 inputs per grid
        #     grid = tcnn.Encoding(2, config)
        #     self.grids.append(grid)


    def set_aabb(self, xyz_max, xyz_min):
        aabb = torch.tensor([
            xyz_max,
            xyz_min
        ])
        self.aabb = nn.Parameter(aabb,requires_grad=True) # !!!!!
        print("Voxel Plane: set aabb=",self.aabb)

    def get_density(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        """Computes and returns the densities."""

        import pdb
        pdb.set_trace()
        pts = normalize_aabb(pts, self.aabb)
        pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]
        pts = pts.reshape(-1, pts.shape[-1])
        
        # get pairings
        # xt = pts[..., [0,3]]
        # yt = pts[..., [1,3]]
        # zt = pts[..., [2,3]]
        # xy = pts[..., [0,1]]
        # yz = pts[..., [1,2]]
        # xz = pts[..., [1,3]]

        # # add axis for num batches
        # if pts.dim() == 2:
        #     coords = coords.unsqueeze(0)
        coord_combs = list(itertools.combinations(
            range(pts.shape[-1]), 2)
        )
        feats = 1.
        for combo, enc in zip(coord_combs, self.grids): #TODO: should only need 1 grid if they all the same anyways
            tcnn_input = pts[...,combo].view(-1, 2)
            tcnn_output = enc(tcnn_input)
            pts_enc = tcnn_output.view(*pts.shape[:-1], tcnn_output.shape[-1])
            feats = feats * pts_enc
            # this is placeholder for now, i really need to compute product of each all 6 planes
            # for each resolution it seems. how does this compute factored product of low rank matrices tho?
            # seems to be formulated differently than original hexplane strategy
            #TODO: are features interpolated using this hashing strategy?

        # if len(feats) < 1:
        #     feats = torch.zeros((0, 1)).to(feats.device)
            
        return feats

    def forward(self,
                pts: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None):
        features = self.get_density(pts, timestamps)

        return features