import torch

class PhotoSplatter(torch.nn):
    """
    """

    def __init__(self, args):
        super().__init__()
        
        # photometric calibration
        self.y = args.gamma
        self.g = args.gain
        self.k = args.cosine_decay

        # gaussian photometric initialization
        self.gaussians = torch.empty(0) # points [N,3]
        self.rotation = torch.empty(0)
        self.scaling = torch.empty(0)
        self.initialize_gaussians()
    
        # delta net
        self.deform_net = DeformNet()

    
    def forward(self, x):
        pass
        #
    

class DeformNet(torch.nn):
    def __init__(self):
        super().__init__()
        # create multires spatial delta feature planes
    
        # create single res temporal delta planes
    
        # create photometric delta planes