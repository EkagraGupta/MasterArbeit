# from sewar import ssim, msssim, uqi
from skimage.metrics import structural_similarity as ssim
import numpy as np
from PIL import Image

import torch

def ssim_operation(im1: Image.Image, im2: Image.Image):
    # im1_np = np.array(im1.convert('L'))
    # im2_np = np.array(im2.convert('L'))
    im1_np = np.array(im1)
    im2_np = np.array(im2)

    # Ensure the win_size is appropriate for your image dimensions
    win_size = min(im1_np.shape[0], im1_np.shape[1], 7)

    ssim_index, _ = ssim(im1_np, im2_np, full=True, channel_axis=2, win_size=win_size)          # color image
    # ssim_index, _ = ssim(im1_np, im2_np, full=True)                                             # grayscale image

    # ssim_index = ssim(im1_np, im2_np)
    return (ssim_index + 1) / 2

# def msssim_operation(im1: Image.Image, im2: Image.Image):
#     im1_np = np.array(im1)
#     im2_np = np.array(im2)

#     msssim_index = msssim(im1_np, im2_np)

#     if isinstance(msssim_index, complex):
#         msssim_index = msssim_index.real

#     return msssim_index

# def uqi_operation(im1: Image.Image, im2: Image.Image):
#     im1_np = np.array(im1)
#     im2_np = np.array(im2)

#     uqi_index = uqi(im1_np, im2_np)

#     return uqi_index

if __name__ == "__main__":
    im1_path = "/home/ekagra/Documents/GitHub/MasterArbeit/example/original_image.png"
    im2_path = "/home/ekagra/Documents/GitHub/MasterArbeit/example/augmented_image_pixelwise.png"
    im1 = Image.open(im1_path)
    im2 = Image.open(im2_path)
    # im1_gray = im1.convert("L")
    # im2_gray = im2.convert("L")

    # matches = sift_operation(im1=im1_gray, im2=im2_gray, display_matches=True)
    sim = ssim_operation(im1, im2)
    print(f"Correction factor: {sim}")