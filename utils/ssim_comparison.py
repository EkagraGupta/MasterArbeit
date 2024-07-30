from skimage.metrics import structural_similarity as ssim
import numpy as np
from PIL import Image

def ssim_operation(im1: Image.Image, im2: Image.Image):
    im1_np = np.array(im1.convert('L'))
    im2_np = np.array(im2.convert('L'))
    ssim_index, _ = ssim(im1_np, im2_np, full=True)
    return (ssim_index + 1) / 2