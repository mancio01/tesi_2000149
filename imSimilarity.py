import numpy as np
import lpips
import preprocessing as pp

from image_similarity_measures import quality_metrics as qm 
from torchvision import transforms
import pickle

loss_fn_alex = lpips.LPIPS(net='alex') 
loss_fn_vgg = lpips.LPIPS(net='vgg')

def imSimilarity(img_x, img_y) :
    t_img_x = transforms.functional.to_tensor(img_x)
    t_img_y = transforms.functional.to_tensor(img_y)
    evaluations = {
                   "rmse": float(qm.ssim(img_x,img_y)),
                   "uiq": float(qm.uiq(img_x,img_y)),
                   "lfa": loss_fn_alex(t_img_x, t_img_y).item(),
                   "lfv": loss_fn_vgg(t_img_x, t_img_y).item()
                   }
    return evaluations




with open('svm.pkl', 'rb') as file:
    svm = pickle.load(file)

with open('svm_o.pkl', 'rb') as file:
    svm_o = pickle.load(file)
    
with open('clustering.pkl', 'rb') as file:
    clus = pickle.load(file)

def areSimilar(img_x, img_y, algorithm='SVM'):
    img_x=pp.preprocess(img_x)
    img_y=pp.preprocess(img_y)
    t_img_x = transforms.functional.to_tensor(img_x)
    t_img_y = transforms.functional.to_tensor(img_y)
    rmse = float(qm.ssim(img_x,img_y))
    uiq = float(qm.uiq(img_x,img_y))
    lfa = loss_fn_alex(t_img_x, t_img_y).item()
    lfv = loss_fn_vgg(t_img_x, t_img_y).item()
    if(algorithm == 'SVM'):
        return svm.predict(np.array([[lfv,uiq,lfa,rmse]]))
    elif(algorithm == 'SVM_o'):
        return svm_o.predict([lfv,uiq,lfa,rmse])
    elif(algorithm == 'spectral_clustering'):
        return clus.predict([lfv,uiq,lfa,rmse])
    raise Exception("invalid algorithm value")     
