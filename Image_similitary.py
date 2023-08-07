import cv2 
#import torch


import numpy as np

def crop_image(img, vertices):
  mask = np.zeros(img.shape[:2], dtype=np.uint8)
  cv2.fillPoly(mask, [np.array(vertices, np.int32)], 255)
  return cv2.bitwise_and(img, img, mask = mask)

def imcompare(img_x, img_y, thr, method):
  if method == "RMSE":
    return RMSE(img_x, img_y) <= thr 
  elif method == "PSNR":
    return PSNR(img_x, img_y) <= thr
  else:
    raise ValueError("Unknown method")
  
def imsimilarity(img_x, img_y, method):
  if method == "RMSE":
    return 1-RMSE(img_x, img_y)
  elif method == "PSNR":
    return 1-PSNR(img_x, img_y)
  else:
    raise ValueError("Unknown method")
  

def RMSE(img_x, img_y):
  rmse = np.sqrt(np.mean((img_x-img_y)**2))/255.0
  return rmse 

def PSNR(img_x, img_y):
  psnr =  cv2.PSNR(img_x, img_y)
  return psnr 

