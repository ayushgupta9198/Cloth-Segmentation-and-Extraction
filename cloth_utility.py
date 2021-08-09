from pylab import imshow
from PIL import Image
import numpy as np
import cv2
import torch
import albumentations as albu
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from cloths_segmentation.pre_trained_models import create_model
import os
import sys


def process_image(image_path:str=None, output_path:str=None )->None:
    """ 
    Process image function take image path as string and save genereated 
    image at output path. 
    """

    model = create_model("Unet_2020-10-30")
    model.eval()
  
    image = load_rgb(image_path)

    transform = albu.Compose([albu.Normalize(p=1)], p=1)
    padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
    x = transform(image=padded_image)["image"]
    x = torch.unsqueeze(tensor_from_rgb_image(x), 0)

    with torch.no_grad():
        prediction = model(x)[0][0]
    mask = (prediction > 0).cpu().numpy().astype(np.uint8)
    mask = unpad(mask, pads)
    
    res1 = cv2.bitwise_and(image,image, mask=mask)
    im_res = Image.fromarray(res1)
    im_res.save(output_path)

def process_dir(INPUT_DIR, OUTPUT_DIR):
    for fname in os.listdir(INPUT_DIR):
        image_path = os.path.join(INPUT_DIR, fname)
        output_path = os.path.join(OUTPUT_DIR, "seg_"+fname)
        process_image(image_path, output_path)



if __name__=='__main__':
    if sys.argv[1] is not None:
        INPUT_DIR = sys.argv[1]
    else:
        print("please provide input dir path")

    if sys.argv[1] is not None:
        OUTPUT_DIR = sys.argv[2]
    else :
        OUTPUT_DIR = "./output"
        
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    process_dir(INPUT_DIR, OUTPUT_DIR)

