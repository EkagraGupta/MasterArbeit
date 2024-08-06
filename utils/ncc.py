import torch
import numpy as np
from PIL import Image
from scipy.signal import correlate2d

# def normalized_cross_correlation(im1: Image.Image, im2: Image.Image):
#     # Convert images to numpy
#     im1_np = np.array(im1.convert('L'))
#     im2_np = np.array(im2.convert('L'))

#     # Normalize the images
#     im1_np = (im1_np - np.mean(im1_np)) / (np.std(im1_np) + 1e-4)
#     im2_np = (im2_np - np.mean(im2_np)) / (np.std(im2_np) + 1e-4)

#     # Compute the numerator and denominator
#     numerator = np.sum(im1_np * im2_np)
#     denominator = np.sqrt(np.sum(im1_np**2) * np.sum(im2_np**2))

#     # Check if the denominator is zero
#     if denominator == 0:
#         print(numerator, denominator)
#         return 0  # or np.nan, depending on how you want to handle this case

#     # Compute the normalized cross-correlation
#     ncc = numerator / denominator
#     return (ncc + 1) / 2


def normalized_cross_correlation(im1: Image.Image, im2: Image.Image):
    # Convert images to grayscale numpy arrays
    im1_np = np.array(im1.convert("L"), dtype=np.float32)
    im2_np = np.array(im2.convert("L"), dtype=np.float32)

    # Normalize the images
    im1_np = (im1_np - np.mean(im1_np)) / (np.std(im1_np) + 1e-4)
    im2_np = (im2_np - np.mean(im2_np)) / (np.std(im2_np) + 1e-4)

    # Compute the normalized cross-correlation using scipy
    ncc_matrix = correlate2d(im1_np, im2_np, mode="valid")
    ncc = np.max(ncc_matrix) / (im1_np.size)

    return (ncc + 1) / 2


if __name__ == "__main__":
    im1 = Image.open(
        "/home/ekagra/Documents/GitHub/MasterArbeit/example/original_image.png"
    )
    im2 = Image.open(
        "/home/ekagra/Documents/GitHub/MasterArbeit/example/augmented_image_geometric.png"
    )
    ncc_score = normalized_cross_correlation(im1, im2)

    print(ncc_score)
