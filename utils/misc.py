import random
import sys
from datetime import datetime
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


def get_proj_matrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


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
    """
    Finds mean squared dist of each point to its nearest 3 neighbors
    """
    points_np = points.detach().cpu().float().numpy()
    dists, inds = KDTree(points_np).query(points_np, k=4)
    meanDists = (dists[:, 1:] ** 2).mean(1)
    return torch.tensor(meanDists, dtype=points.dtype, device=points.device)


def strip_symmetric(L):
    """
    Takes Nx3x3 matrix and saves upper triangle of each row, creating Nx6 output.
    """
    # Generally used for covariance, so variable termed uncertainty
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def convert_quat2rot(r):
    """
    Takes in quaternions, normalizes them, and then converts them to 3x3 rotation matrices.
    """
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    q = r / norm[:, None]
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    
    R = torch.zeros((q.size(0), 3, 3), device='cuda')
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = convert_quat2rot(r)

    # create scales by putting them on diag of matrix
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    # scale rotation by multiplying it by scaling matrix
    L = R @ L
    return L


def normalize_aabb(pts, aabb):
    """
    Normalize axis-aligned bounding box around current pts
    """
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


def initialize_weights(m):
    """
    Initialize network weights.
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            torch.nn.init.xavier_uniform_(m.weight,gain=1)


def print_gpu_memory_usage():
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f"Memory Allocated: {allocated / 1e6:.2f} MB")
    print(f"Memory Reserved: {reserved / 1e6:.2f} MB")