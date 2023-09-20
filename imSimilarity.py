import lpips
from image_similarity_measures import quality_metrics as qm 
from torchvision import transforms

loss_fn_alex = lpips.LPIPS(net='alex') 
loss_fn_vgg = lpips.LPIPS(net='vgg')

def imSimilarity(img_x, img_y) :
    t_img_x = transforms.functional.to_tensor(img_x)
    t_img_y = transforms.functional.to_tensor(img_y)
    evaluations = {
                   "ssim": float(qm.ssim(img_x,img_y)),
                   "uiq": float(qm.uiq(img_x,img_y)),
                   "lfa": loss_fn_alex(t_img_x, t_img_y).item(),
                   "lfv": loss_fn_vgg(t_img_x, t_img_y).item()
                   }
    return evaluations