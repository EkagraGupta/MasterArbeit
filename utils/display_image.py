import matplotlib.pyplot as plt

def get_images(original_image, augmented_image):
    
    # Convert images to displayable format
    original_image_np = original_image[0].permute(1, 2, 0).numpy()  # Convert CHW to HWC format
    augmented_image_np = augmented_image[0].permute(1, 2, 0).numpy()  # Convert CHW to HWC format

    # Plot the original and augmented images
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    axs[0].imshow(original_image_np)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    axs[1].imshow(augmented_image_np)
    axs[1].set_title('Augmented Image')
    axs[1].axis('off')
    
    plt.show()