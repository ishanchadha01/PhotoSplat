import random
import sys
import datetime
import time
from typing import NamedTuple

import torch
import numpy as np


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
    Get w2c transformation, currently given as c2w
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