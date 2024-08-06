from sewar import mse, rmse, psnr, ssim, uqi, ergas, scc, rase, sam, msssim
from PIL import Image
import time
import numpy as np

def compare_images(im1, im2):
    # Convert images to numpy arrays
    im1_np = np.array(im1)
    im2_np = np.array(im2)

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'PSNR': psnr,
        'SSIM': ssim,
        'UQI': uqi,
        'ERGAS': ergas,
        'SCC': scc,
        'RASE': rase,
        'SAM': sam,
        'MSSSIM': msssim
    }

    results = {}
    for name, func in metrics.items():
        start_time = time.time()
        result = func(im1_np, im2_np)
        end_time = time.time()
        time_taken = end_time - start_time
        results[name] = (result, time_taken)

    return results

if __name__=='__main__':
    im1_path = '/home/ekagra/Documents/GitHub/MasterArbeit/example/original_image.png'
    im2_path = '/home/ekagra/Documents/GitHub/MasterArbeit/example/augmented_image_pixelwise.png'

    im1 = Image.open(im1_path)
    im2 = Image.open(im2_path)

    comparison_results = compare_images(im1, im2)

    for metric, (value, duration) in comparison_results.items():
        print(f"{metric}: {value}, Time taken: {duration:.4f} seconds")