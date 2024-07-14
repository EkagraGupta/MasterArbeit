import torch
import numpy as np
from PIL import Image


def normalized_cross_correlation(im1: torch.tensor, im2: torch.tensor):
    # Convert images to numpy
    im1_np = im1.numpy()
    im2_np = im2.numpy()

    # Normalize the images
    im1_np = (im1_np - np.mean(im1_np)) / np.std(im1_np)
    im2_np = (im2_np - np.mean(im2_np)) / np.std(im2_np)

    # Compute the normalized cross-correlation
    ncc = np.sum(im1_np * im2_np) / np.sqrt(np.sum(im1_np**2) * np.sum(im2_np**2))
    return ncc


if __name__ == "__main__":
    im1 = Image.open("/home/ekagra/Desktop/Study/MA/code/example/original_image.png")
    im2 = Image.open("/home/ekagra/Desktop/Study/MA/code/example/augmented_image.png")

    im1_tensor = torch.tensor(np.array(im1).astype(np.float32)).permute(2, 0, 1)
    im2_tensor = torch.tensor(np.array(im2).astype(np.float32)).permute(2, 0, 1)

    ncc_score = normalized_cross_correlation(im1_tensor, im2_tensor)

    print(ncc_score)
