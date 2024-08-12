import numpy as np
from PIL import Image
import torch
import cv2
from scipy.signal import correlate2d
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from psnr_hvsm.torch import psnr_hvs_hvsm, bt601ycbcr
from torchmetrics.image import StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure

def normalized_cross_correlation(im1: Image.Image, im2: Image.Image):
    # convert images to grayscale numpy arrays
    im1_np = np.array(im1.convert("L"), dtype=np.float32)
    im2_np = np.array(im2.convert("L"), dtype=np.float32)

    # normalize the images
    im1_np = (im1_np - np.mean(im1_np)) / (np.std(im1_np) + 1e-4)
    im2_np = (im2_np - np.mean(im2_np)) / (np.std(im2_np) + 1e-4)

    # compute the normalized cross correlation
    ncc_matrix = correlate2d(im1_np, im2_np, mode="valid")
    ncc_value = np.max(ncc_matrix) / (im1_np.size)

    return (ncc_value + 1) / 2


def structural_similarity(im1: Image.Image, im2: Image.Image):
    # convert images to numpy arrays
    im1_np = np.array(im1)
    im2_np = np.array(im2)

    # compute the structural similarity index
    ssim_index, _ = ssim(im1_np, im2_np, full=True, channel_axis=2)

    return (ssim_index + 1) / 2


def psnr_hvs_calculation(im1: Image.Image, im2: Image.Image, scaling_factor: int=100):
    if not isinstance(im1, torch.Tensor) or not isinstance(im2, torch.Tensor):
        to_tensor = transforms.ToTensor()
        im1 = to_tensor(im1)
        im2 = to_tensor(im2)

    im1_y, *_ = bt601ycbcr(im1)
    im2_y, *_ = bt601ycbcr(im2)

    psnr_hvs, psnr_hvsm = psnr_hvs_hvsm(im1_y, im2_y)
    psnr_normalized = psnr_hvsm / scaling_factor
    return min(psnr_normalized, 1.0)

def histogram_comparison(im1: Image.Image, im2: Image.Image):
    im1_np = np.array(im1)
    im2_np = np.array(im2)

    im1_hist = cv2.calcHist([im1_np], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    im1_hist[255, 255, 255] = 0
    cv2.normalize(im1_hist, im1_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    im2_hist = cv2.calcHist([im2_np], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    im2_hist[255, 255, 255] = 0
    cv2.normalize(im2_hist, im2_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    metric_val = cv2.compareHist(im1_hist, im2_hist, cv2.HISTCMP_CORREL)
    # print(f'Histogram Comparison Value: {metric_val:.3f}')
    return metric_val

def multiscale_structural_similarity(im1: Image.Image, im2: Image.Image):
    # convert images to numpy arrays
    # im1_np = np.array(im1)
    # im2_np = np.array(im2)
    to_tensor = transforms.ToTensor()
    im1 = to_tensor(im1).unsqueeze(0)
    im2 = to_tensor(im2).unsqueeze(0)
    # compute the structural similarity index
    sim = StructuralSimilarityIndexMeasure()
    sim_val = sim(im1, im2)
    return sim_val.item()

if __name__ == "__main__":
    im1_path = "/home/ekagra/Documents/GitHub/MasterArbeit/example/original_image.png"
    im2_path = "/home/ekagra/Documents/GitHub/MasterArbeit/example/original_image.png"
    im1 = Image.open(im1_path)
    im2 = Image.open(im2_path)
    
    to_tensor = transforms.ToTensor()
    # ncc_value = normalized_cross_correlation(im1, im2)
    ssim_value = structural_similarity(im1, im2)
    print(ssim_value)
    # psnr_value = psnr_hvs_calculation(im1, im2)
    # hist_value = histogram_comparison(im1, im2)
    # print(f'Normalized Cross Correlation Value: {ncc_value}\nSSIM: {ssim_value}\nPSNR Value: {psnr_value}\nHistogram Comparison Value: {hist_value}')
    multiscale_structural_similarity(to_tensor(im1), to_tensor(im2))