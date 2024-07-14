import numpy as np
from skimage import io, color
from skimage.metrics import structural_similarity as ssim       # another metric

# from skimage.measure import compare_vif
import piq
import torch

im1 = io.imread("/home/ekagra/Desktop/Study/MA/code/example/original_image.png")
im2 = io.imread("/home/ekagra/Desktop/Study/MA/code/example/augmented_image.png")
# Convert images to PyTorch tensors
im1_tensor = torch.tensor(im1).float().permute(2, 0, 1).unsqueeze(0)
im2_tensor = torch.tensor(im2).float().permute(2, 0, 1).unsqueeze(0)
im1_tensor /= 255.0
im2_tensor /= 255.0
vif_value = piq.vif_p(im1_tensor, im2_tensor)
print(im1_tensor.shape)
print(vif_value)
