import preprocessing as pp
import os
import json

import lpips
import torch
from image_similarity_measures import quality_metrics as qm 
from torchvision import transforms


with open(os.path.dirname(os.path.abspath(__file__)) + r"\ratings.json", "r") as file:
    data = json.load(file)
ratings = data["train"]
true = []
false = []
sonda = 0
path = os.path.dirname(os.path.abspath(__file__)) + r"\filtering\train\\"
folders = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
loss_fn_alex = lpips.LPIPS(net='alex') 
loss_fn_vgg = lpips.LPIPS(net='vgg')
t_img_x = torch.zeros(224,224)
t_img_y = torch.zeros(224,224)
for i in folders:
    for j in os.listdir(os.path.dirname(os.path.abspath(__file__)) + r"\filtering\train\\"+i):
        if j.endswith("first.png"):
            img_x = pp.preprocess(path + i + "\\" + j)
            t_img_x = transforms.functional.to_tensor(img_x)   
        elif j.endswith("second.png"):
            img_y = pp.preprocess(path + i + "\\" + j)
            t_img_y = transforms.functional.to_tensor(img_y)
    evaluations = {"rmse": float(qm.rmse(img_x,img_y)),
                   "psnr": float(qm.psnr(img_x,img_y)),
                   "ssim": float(qm.ssim(img_x,img_y)),
                   "fsim": float(qm.fsim(img_x,img_y)),
                   "sre": float(qm.sre(img_x,img_y)),
                   "sam": float(qm.sam(img_x,img_y)),
                   "uiq": float(qm.uiq(img_x,img_y)),
                   "lfa": loss_fn_alex(t_img_x, t_img_y).item(),
                   "lfv": loss_fn_vgg(t_img_x, t_img_y).item()
                   }
    if(ratings[i]):
        true.append(evaluations)
    elif(not ratings[i]):
        false.append(evaluations) 
    print(sonda)
    sonda = sonda + 1

data = { True : true, False : false}
with open(os.path.dirname(os.path.abspath(__file__)) + r"\index.json", 'w') as json_file:
    json.dump(data, json_file, indent=4)

