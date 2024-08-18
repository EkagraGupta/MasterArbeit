import numpy as np
from PIL import Image
import torch
import cv2
from scipy.signal import correlate2d
from skimage.metrics import structural_similarity
from torchvision import transforms
# from psnr_hvsm.torch import psnr_hvs_hvsm, bt601ycbcr
from torchmetrics.image import VisualInformationFidelity, StructuralSimilarityIndexMeasure, SpatialCorrelationCoefficient, LearnedPerceptualImagePatchSimilarity, PeakSignalNoiseRatio, RelativeAverageSpectralError, PeakSignalNoiseRatioWithBlockedEffect, SpectralAngleMapper, UniversalImageQualityIndex


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


def structural_similarity_calculation(im1: Image.Image, im2: Image.Image):
    # convert images to numpy arrays
    im1_np = np.array(im1)
    im2_np = np.array(im2)

    # compute the structural similarity index
    # ssim_index, _ = structural_similarity(im1_np, im2_np, full=True, channel_axis=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0)
    ssim_index, _ = structural_similarity(im1=im1_np,
                                          im2=im2_np,
                                          gaussian_weights=True,
                                          sigma=1.5,
                                          use_sample_covariance=False,
                                          full=True,
                                          channel_axis=2,
                                          data_range=1.0)

    return ssim_index


# def psnr_hvs_calculation(im1: Image.Image, im2: Image.Image, scaling_factor: int = 100):
#     if not isinstance(im1, torch.Tensor) or not isinstance(im2, torch.Tensor):
#         to_tensor = transforms.ToTensor()
#         im1 = to_tensor(im1)
#         im2 = to_tensor(im2)

#     im1_y, *_ = bt601ycbcr(im1)
#     im2_y, *_ = bt601ycbcr(im2)

#     psnr_hvs, psnr_hvsm = psnr_hvs_hvsm(im1_y, im2_y)
#     psnr_normalized = psnr_hvsm / scaling_factor
#     return min(psnr_normalized, 1.0)


def histogram_comparison(im1: Image.Image, im2: Image.Image):
    im1_np = np.array(im1)
    im2_np = np.array(im2)

    im1_hist = cv2.calcHist([im1_np], [0, 1, 2], None, [
                            256, 256, 256], [0, 256, 0, 256, 0, 256])
    im1_hist[255, 255, 255] = 0
    cv2.normalize(im1_hist, im1_hist, alpha=0,
                  beta=1, norm_type=cv2.NORM_MINMAX)
    im2_hist = cv2.calcHist([im2_np], [0, 1, 2], None, [
                            256, 256, 256], [0, 256, 0, 256, 0, 256])
    im2_hist[255, 255, 255] = 0
    cv2.normalize(im2_hist, im2_hist, alpha=0,
                  beta=1, norm_type=cv2.NORM_MINMAX)

    metric_val = cv2.compareHist(im1_hist, im2_hist, cv2.HISTCMP_CORREL)
    # print(f'Histogram Comparison Value: {metric_val:.3f}')
    return metric_val


def multiscale_structural_similarity(im1: Image.Image, im2: Image.Image):
    to_tensor = transforms.ToTensor()
    im1 = to_tensor(im1).unsqueeze(0)
    im2 = to_tensor(im2).unsqueeze(0)

    # compute the structural similarity index
    ssim = StructuralSimilarityIndexMeasure(return_contrast_sensitivity=True)
    structural_value, contrast_value = ssim(im1, im2)
    luminance_value = structural_value / contrast_value
    # print(f'Structural Value: {structural_value:.3f}\tContrast Value: {contrast_value:.3f}\tLuminance Value: {luminance_value:.3f}')
    # return structural_value.item()
    return luminance_value.item()


def spatial_correlation_coefficient(im1: Image.Image, im2: Image.Image):
    to_tensor = transforms.ToTensor()
    im1 = to_tensor(im1).unsqueeze(0)
    im2 = to_tensor(im2).unsqueeze(0)

    # compute the spatial correlation coefficient
    scc = SpatialCorrelationCoefficient()
    spatial_correlation_value = scc(im1, im2)
    return spatial_correlation_value.item()


# DNT: too slow, uses alexnet/VGG models for comparison
def learned_perceptual_similarity(im1: Image.Image, im2: Image.Image):
    to_tensor = transforms.ToTensor()
    im1 = to_tensor(im1).unsqueeze(0)
    im2 = to_tensor(im2).unsqueeze(0)

    # compute the learned perceptual image patch similarity
    lpips = LearnedPerceptualImagePatchSimilarity()
    perceptual_similarity_value = lpips(im1, im2)
    return perceptual_similarity_value.item()


def peak_signal_noise_ratio(im1: Image.Image, im2: Image.Image, max_psnr=100.0):
    to_tensor = transforms.ToTensor()
    im1 = to_tensor(im1).unsqueeze(0)
    im2 = to_tensor(im2).unsqueeze(0)

    # compute the peak signal-to-noise ratio
    psnr = PeakSignalNoiseRatio(base=10)
    psnr_value = psnr(im1, im2)

    # normalize psnr_value
    psnr_value = psnr_value / max_psnr
    return min(psnr_value.item(), 1.0)


def peak_signal_noise_ratio_blocked(im1: Image.Image, im2: Image.Image, max_psnr=100.0):
    to_tensor = transforms.ToTensor()
    im1 = to_tensor(im1).unsqueeze(0)
    im2 = to_tensor(im2).unsqueeze(0)

    # compute the peak signal-to-noise with blocked effect ratio
    psnr_b = PeakSignalNoiseRatioWithBlockedEffect(block_size=8)
    psnr_b_value = psnr_b(im1, im2)

    # normalize psnr_b_value
    psnr_b_value = psnr_b_value / max_psnr
    return min(psnr_b_value.item(), 1.0)


# DNT: there are no grounds to normalize it
def relative_average_spectral_error(im1: Image.Image, im2: Image.Image):
    to_tensor = transforms.ToTensor()
    im1 = to_tensor(im1).unsqueeze(0)
    im2 = to_tensor(im2).unsqueeze(0)

    # compute the relative average spectral error
    rase = RelativeAverageSpectralError()
    spectral_error_value = rase(im1, im2)
    return spectral_error_value.item()


# have to debug, does not work
def spectral_angle_mapper(im1: Image.Image, im2: Image.Image):
    to_tensor = transforms.ToTensor()
    im1 = to_tensor(im1).unsqueeze(0)
    im2 = to_tensor(im2).unsqueeze(0)

    # compute the spectral angle mapper
    sam = SpectralAngleMapper()
    spectral_angle_value = sam(im1, im2)
    return 1.0 - spectral_angle_value.item()


def universal_image_quality_index(im1: Image.Image, im2: Image.Image):
    to_tensor = transforms.ToTensor()
    im1 = to_tensor(im1).unsqueeze(0)
    im2 = to_tensor(im2).unsqueeze(0)

    # compute the universal image quality index
    uiq = UniversalImageQualityIndex()
    uiq_value = uiq(im1, im2)
    return uiq_value.item()


# too slow and resize required
def visual_information_fidelity(im1: Image.Image, im2: Image.Image):
    resize = transforms.Resize(41)
    to_tensor = transforms.ToTensor()
    im1 = to_tensor(resize(im1)).unsqueeze(0)
    im2 = to_tensor(resize(im2)).unsqueeze(0)

    # compute the visual information fidelity
    vif = VisualInformationFidelity()
    vif_value = vif(im1, im2)
    return vif_value.item()

def gaussian(x, a, b, c):
    gauss = a * np.exp(-0.5 * ((x - b) / c) ** 2)
    if np.any(gauss>1.0):
        gauss.where(gauss>1.0, 1.0, inplace=True)
    return gauss

def custom_function(x, a, b, c, d, e):
    # best values: [ 1.2438093   7.18937766 -0.87255438 -0.0573816  -0.2456411 ]
    result = a / (1.0 + np.exp(-b * (x - c))) + d * x + e
    if np.any(result>1.0):
        result.where(result>1.0, 1.0, inplace=True)
        
    return result

def sigmoid(x, a, b, c):
    result = a / (1.0 + np.exp(-b * (x - c)))
    if np.any(result>1.0):
        result.where(result>1.0, 1.0, inplace=True)
    return result


if __name__ == "__main__":
    im1_path = "/home/ekagra/Documents/GitHub/MasterArbeit/example/original_image.png"
    im2_path = "/home/ekagra/Documents/GitHub/MasterArbeit/example/augmented_image.png"
    im1 = Image.open(im1_path)
    im2 = Image.open(im2_path)
