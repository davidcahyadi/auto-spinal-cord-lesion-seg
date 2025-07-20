import torch 
import cv2 
from PIL import Image 
import numpy as np 

def combine_mask(image, mask):
    if len(mask.shape) > 2:
        mask = mask.squeeze()
    image = (image * 255).int()

    if image.shape[-1] == 1:
        image = torch.cat([image, image, image], dim=2).cpu().numpy()
    else:
        image = image.cpu().numpy()
    if mask.shape[-1] == 3:
        mask = cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
    image[mask == 1] = np.array([255, 255, 0])

    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    return image