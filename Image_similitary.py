import cv2 

import numpy as np

def sim_RMSE(img_x, img_y):
  rmse = 1 - np.sqrt(np.mean(np.square(img_a-img_b)))/255.0
  return rmse

