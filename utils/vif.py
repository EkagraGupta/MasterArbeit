import piq
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


def compute_vif(im1, im2):
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize(41)

    if im1.size[0] < 41 or im2.size[0] < 41:
        im1 = resize(im1)
        im2 = resize(im2)

    im1_tensor = to_tensor(im1)
    im2_tensor = to_tensor(im2)

    if len(im1_tensor.shape) < 4 or len(im2_tensor.shape) < 4:
        im1_tensor = im1_tensor.unsqueeze(0)
        im2_tensor = im2_tensor.unsqueeze(0)

    vif_value = piq.vif_p(im1_tensor, im2_tensor)
    return np.clip(vif_value, 0.0, 1.0)


if __name__ == "__main__":
    im1_path = "/home/ekagra/Documents/GitHub/MasterArbeit/example/original_image.png"
    im2_path = "/home/ekagra/Documents/GitHub/MasterArbeit/example/augmented_image_pixelwise.png"
    im1 = Image.open(im1_path)
    im2 = Image.open(im2_path)

    vif_val = compute_vif(im1, im2)
    print(vif_val)

    # Plot the images side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(im1)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(im2)
    axes[1].set_title('Augmented Image')
    axes[1].axis('off')

    # Set the VIF value as the title of the figure
    fig.suptitle(f'Visual Information Fidelity (VIF): {vif_val.item():.4f}', fontsize=16)

    plt.show()
