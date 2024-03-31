import pytest
import numpy as np

from utils.misc import get_world2view

def test_dataset_creation():
    assert True

def test_guassian_colmap_initialization():
    assert True

# TODO: add data fixtures
def test_get_world2view2():
    x = np.array([
        [0.866, -0.5, 0, 3],
        [0.5, 0.866, 0, 2],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    R = x[:3,:3]
    t=x[:3,3]
    out = get_world2view(R, t, translate=np.array([.1, .1, .2]), scale=1.5)
    expected = np.array([[0.866, 0.5, 0., 4.294902],
        [-0.5, 0.866, 0., 2.944968],
        [0., 0., 1., -0.3],
        [0., 0., 0., 1.]]
    )
    assert np.all(out == expected)