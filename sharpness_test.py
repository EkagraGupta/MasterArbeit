import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F

# Load an example image
image = Image.open('example/original_image.png')

# Define a range of sharpness factors and sigma values for comparison
sharpness_factors = np.linspace(-1, 0, 10)  # Adjust sharpness from 0 (max blur) to 1 (no change)
sigma_values = np.linspace(0.1, 10, 10)  # Corresponding sigma values for GaussianBlur

# Set up plots
fig, axes = plt.subplots(2, 10, figsize=(18, 10))
fig.suptitle('Comparison of adjust_sharpness vs GaussianBlur', fontsize=16)

# Apply adjust_sharpness and GaussianBlur, then display side by side
for i, (sharpness, sigma) in enumerate(zip(sharpness_factors, sigma_values)):
    # Apply adjust_sharpness
    sharpened_img = F.adjust_sharpness(image, sharpness_factor=1.0 + sharpness)
    # Apply Gaussian blur
    blurred_img = F.gaussian_blur(image, kernel_size=(5, 5), sigma=sigma)
    
    # Plot sharpened image
    axes[0, i].imshow(sharpened_img)
    axes[0, i].set_title(f'Sharpness: {sharpness:.2f}')
    axes[0, i].axis('off')
    
    # Plot Gaussian blurred image
    axes[1, i].imshow(blurred_img)
    axes[1, i].set_title(f'Sigma: {sigma:.2f}')
    axes[1, i].axis('off')

# Display the plots
plt.tight_layout()
plt.show()
