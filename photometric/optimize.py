import numpy as np
from tqdm import tqdm
import cv2

from utils import get_calibration, unproject_camera_model, get_intensity,\
      calibrated_photometric_endoscope_model, cost_function, regularization_function

def compute_img_depths(img, k, g_t, gamma, iters=1000):
    """
    x_l: light center
    x_i: point on surface
    mu: light spread func
    """

    # set color img to 0-1 intensity range
    img = 1/255 * img

    # initialize energy and gradient
    energy_function = np.zeros((img.shape[0], img.shape[1]))
    gradient = np.zeros((img.shape[0], img.shape[1]))
    depth_map = np.zeros((img.shape[0], img.shape[1]))
    errors = []

    # Minimize energy function
    regularization_lambda = 0.5 # adjust
    alpha = 5 # adjust learning rate
    prev_energy_function = np.zeros((img.shape[0], img.shape[1]))
    prev_depth_map = np.zeros((img.shape[0], img.shape[1]))

    for i in tqdm(range(iters)):
        # Compute energy function for every pixel separately
        for row in tqdm(range(img.shape[0])):
            for col in range(img.shape[1]):
                d = depth_map[row, col]
                u = img[row, col] # pixel
                x, y, z = unproject_camera_model(u, d) # angle to use in photometric model
                L = calibrated_photometric_endoscope_model(x, y, d, k, g_t, gamma) # TODO: Why is z never used??
                I = get_intensity(u)
                C = cost_function(I, L)
                R = regularization_function(gradient[row, col])
                energy_function[row, col] = C + regularization_lambda * R

                # Perform gradient descent for every pixel
                if i==0:
                    depth_map[row, col] += .1
                else:
                    gradient[row, col] = energy_function[row, col] - prev_energy_function[row, col]
                    depth_dir = np.sign(depth_map[row, col] - prev_depth_map[row, col])
                    depth_map[row, col] -= depth_dir * alpha * gradient[row, col]
                prev_energy_function[row, col] = energy_function[row, col]
                prev_depth_map[row, col] = depth_map[row, col]

        # Save energy function, depth map, and gradient
        error = np.sum(energy_function)
        print(f"Error: {error}")
        errors.append(error)
        with open("./errors.txt", "w") as f:
            for error in errors:
                f.write(str(error) + "\n")

        # Paper used trust region, this implementation just uses line search (gradient descent)
        # print(gradient[200][200])
        # hessian = compute_hessian(energy_function)
        # trust_region_subproblem(energy_function, gradient, hessian)
        depth_map_img = cv2.normalize(src=depth_map, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        cv2.imwrite(f'../images/depthmap_{i}.png', depth_map_img)

    return depth_map