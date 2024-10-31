from psnr_hvsm.torch import psnr, psnr_hvs_hvsm, bt601ycbcr
from PIL import Image
from torchvision import transforms
import numpy as np
import torch


def psnr_operation(im1, im2, scaling_factor=100.0):
    if not isinstance(im1, torch.Tensor) or not isinstance(im2, torch.Tensor):
        to_tensor = transforms.ToTensor()
        im1 = to_tensor(im1)
        im2 = to_tensor(im2)

    im1_y, *_ = bt601ycbcr(im1)
    im2_y, *_ = bt601ycbcr(im2)

    psnr_hvs, psnr_hvsm = psnr_hvs_hvsm(im1_y, im2_y)
    psnr_hvs = 10**((psnr_hvs / 10)) / 10**((scaling_factor / 10))
    psnr_hvsm = 10**((psnr_hvsm / 10)) / 10**((scaling_factor / 10))
    print(f"PSNR HVS: {psnr_hvs}\nPSNR HVSM: {psnr_hvsm}")
    psnr_normalized = psnr_hvsm / scaling_factor
    return min(psnr_normalized, 1.0)
    # return psnr_hvs


if __name__ == "__main__":
    im1_path = "/home/ekagra/Documents/GitHub/MasterArbeit/example/original_image.png"
    im2_path = "/home/ekagra/Documents/GitHub/MasterArbeit/example/augmented_image_geometric.png"
    im1 = Image.open(im1_path)
    im2 = Image.open(im2_path)

    psnr_value = psnr_operation(im1, im2)
    print(f"PSNR Value: {psnr_value}")
