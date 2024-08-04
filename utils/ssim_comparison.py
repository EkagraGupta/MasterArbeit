from skimage.metrics import structural_similarity as sim
import numpy as np
from PIL import Image

from torchmetrics.functional import structural_similarity_index_measure as ssim
import torch

def ssim_operation(im1: Image.Image, im2: Image.Image):
    # im1_np = np.array(im1.convert('L'))
    # im2_np = np.array(im2.convert('L'))
    im1_np = np.array(im1)
    im2_np = np.array(im2)

    # Ensure the win_size is appropriate for your image dimensions
    win_size = min(im1_np.shape[0], im1_np.shape[1], 7)

    ssim_index, _ = sim(im1_np, im2_np, full=True, channel_axis=2, win_size=win_size)          # color image
    # ssim_index, _ = ssim(im1_np, im2_np, full=True)                                             # grayscale image
    return (ssim_index + 1) / 2

def ssim_operation2(im1: Image.Image, im2: Image.Image):
    im1_np = np.array(im1).astype(np.float32) / 255.0
    im2_np = np.array(im2).astype(np.float32) / 255.0

    # Convert numpy arrays to PyTorch tensors
    im1_tensor = torch.tensor(im1_np).permute(2, 0, 1).unsqueeze(0)  # Shape: (1, C, H, W)
    im2_tensor = torch.tensor(im2_np).permute(2, 0, 1).unsqueeze(0)  # Shape: (1, C, H, W)

    # Calculate SSIM using torchmetrics
    similarity = ssim(im1_tensor, im2_tensor, data_range=1.0)
    return similarity.item()

if __name__ == "__main__":
    im1_path = "/home/ekagra/Documents/GitHub/MasterArbeit/example/original_image.png"
    im2_path = "/home/ekagra/Documents/GitHub/MasterArbeit/example/augmented_image.png"
    im1 = Image.open(im1_path)
    im2 = Image.open(im2_path)
    # im1_gray = im1.convert("L")
    # im2_gray = im2.convert("L")

    # matches = sift_operation(im1=im1_gray, im2=im2_gray, display_matches=True)
    sim = ssim_operation(im1, im2)
    print(f"Correction factor: {sim}")