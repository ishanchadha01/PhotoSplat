import random
import sys
import datetime
import time
from typing import NamedTuple
import math

import torch
import numpy as np
from scipy.spatial import KDTree


class PointCloud(NamedTuple):
    pts: torch.Tensor
    colors: torch.Tensor
    normals: torch.Tensor


class Timer:
    def __init__(self):
        self.start_time = None
        self.elapsed = 0
        self.paused = False

    def start(self):
        if self.start_time is None:
            self.start_time = time.time()
        elif self.paused:
            self.start_time = time.time() - self.elapsed
            self.paused = False

    def pause(self):
        if not self.paused:
            self.elapsed = time.time() - self.start_time
            self.paused = True

    def get_elapsed_time(self):
        if self.paused:
            return self.elapsed
        else:
            return time.time() - self.start_time


def create_seed(seed):
    """
    Set seed for experiments
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_stdout_state(silent):
    """
    Create new file-like object that appends time to every like of output
    """
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


def get_world2view(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    """
    Get w2c transformation, currently given as c2w rotation and w2c translatioin
    """
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.T

    # translate and scale cam center, and then put it back in original coords
    cam_center = -R @ t
    cam_center = (cam_center + translate) * scale
    Rt[:3, 3] = -R.T @ cam_center

    Rt[3,3] = 1.0
    return np.float32(Rt)


def inverse_sigmoid(x):
    return torch.log(x/(1-x))


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def get_focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def distCUDA2(points):
    points_np = points.detach().cpu().float().numpy()
    dists, inds = KDTree(points_np).query(points_np, k=4)
    meanDists = (dists[:, 1:] ** 2).mean(1)
    return torch.tensor(meanDists, dtype=points.dtype, device=points.device)