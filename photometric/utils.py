import numpy as np


def light_spread_function(x, k):
  # TODO: can try changing radians to degrees
  return np.power(np.abs(np.cos(x)), k) # TODO: might need to preserve sign


def calibrated_photometric_endoscope_model(x, y, z, k, g_t, gamma):
  mu_prime = light_spread_function(z, k)
  f_r_theta = 1/np.pi # Lambertian BRDF, might work better with value closer to 0 like 1/2pi
  xc_to_pixel = np.linalg.norm(np.array([x, y, z])) # Find distance from center of image to pixel
  theta = 2 * (np.arccos(np.linalg.norm(np.array([x,y])) / xc_to_pixel)) # Compute angle of incidence, and then find angle theta
  L = (mu_prime / xc_to_pixel) * f_r_theta * np.cos(theta) * g_t
  L  = np.power(np.abs(L), gamma) # TODO: might need to preserve sign
  return L


def cost_function(I, L, thresh=1e-4):
  # Huber norm of pixel intensity and photometric model
  if np.linalg.norm(I-L) <= thresh:
    norm = np.square(I-L) / (2*thresh)
  else:
    norm = np.abs(I-L) + (thresh/2)
  return norm
  

def regularization_function(grad, thresh=1e-4):
  #TODO: why is this always 0???
  g = np.exp(-np.linalg.norm(grad)) # Can adjust mutliplier of gradient (alpha) order or norm (beta)
  # Compute Huber norm of gradient
  if np.linalg.norm(grad) <= thresh:
    norm = np.power(np.linalg.norm(grad), 2) / (2*thresh)
  else:
    norm = np.abs(grad) + (thresh/2)
  return g * norm


def get_intensity(pixel):
  r,g,b = pixel
  return (float(r) + float(g) + float(b)) / (255*3)


def get_camera_params():
  # returns cx, cy, fx, fy and distortion coefficients k1-4 for Kannala Brandt projection model
  return 735.37, 552.80, 717.21, 717.48, -0.13893, -1.2396e-3, 9.1258e-4, -4.0716e-5


def unproject_camera_model(u, d):
  # Kannala Brandt unprojection
  # map from 2D pixel value u to 3D point in world
  cx, cy, fx, fy, k1, k2, k3, k4 = get_camera_params()
  mx = (u[0] - cx) / fx
  my = (u[1] - cy) / fy
  r = (mx**2 + my**2) ** 0.5
  
  # Find roots for model k4(x^9) + k3(x^7) + k2(x^5) + k1(x^3) + x = r where x=theta
  coeffs = np.array([k4, 0, k3, 0, k2, 0, k1, 0, 1, -r])
  roots = np.roots(coeffs)
  theta = np.real(roots[0])
  
  # Get X,Y,Z output
  X = np.sin(theta) * (mx/r)
  Y = np.sin(theta) * (my/r)
  Z = np.cos(theta)
  return mx, my, theta


def get_calibration(img):
  return 2.5, 2.0, 2.2