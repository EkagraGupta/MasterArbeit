import torch
import numpy as np
from PIL import Image


def normalized_cross_correlation(im1: Image.Image, im2: Image.Image):
    # Convert images to numpy
    im1_np = np.array(im1.convert('L'))
    im2_np = np.array(im2.convert('L'))

    # Normalize the images
    im1_np = (im1_np - np.mean(im1_np)) / np.std(im1_np)
    im2_np = (im2_np - np.mean(im2_np)) / np.std(im2_np)
    # Compute the normalized cross-correlation
    ncc = np.sum(im1_np * im2_np) / np.sqrt(np.sum(im1_np**2) * np.sum(im2_np**2))
    return (ncc + 1) / 2


if __name__ == "__main__":
    im1 = Image.open("/home/ekagra/Documents/GitHub/MasterArbeit/example/augmented_image.png")
    im2 = Image.open("/home/ekagra/Documents/GitHub/MasterArbeit/example/original_image.png")
    ncc_score = normalized_cross_correlation(im1, im2)

    print(ncc_score)
