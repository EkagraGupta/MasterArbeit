from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision import transforms

from dataset import load_dataset

def display_sample(original_loader, augmented_loader):
    fig, axes = plt.subplots(2, 1, figsize=(15, 6))

    for i, ((orig_image, orig_label), (aug_image, aug_label)) in enumerate(zip(original_loader, augmented_loader)):
        # Display original image
        orig_image = torchvision.utils.make_grid(orig_image)
        orig_image_numpy = orig_image.numpy()
        axes[0].imshow(np.transpose(orig_image_numpy, (1, 2, 0)))
        axes[0].set_title(f'Original')
        axes[0].axis('off')
        
        # Display augmented image
        aug_image = torchvision.utils.make_grid(aug_image)
        aug_image_numpy = aug_image.numpy()
        axes[1].imshow(np.transpose(aug_image_numpy, (1, 2, 0)))
        axes[1].set_title(f'Augmented')
        axes[1].axis('off')
        break
    
    plt.show()

if __name__ == '__main__':
    batchSize = 1
    originalTransform = transforms.Compose([transforms.Resize((224, 224)),
                                                transforms.ToTensor()])
    augmentedTransform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                                transforms.RandomCrop(100),
                                                transforms.TrivialAugmentWide(),
                                                #transforms.Lambda(lambda x: x.mul(255).byte()),
                                                transforms.ToTensor()
                                                ])

    _, originalTestloader, _ = load_dataset(batch_size=batchSize, transform=originalTransform)
    _, augmentedTestloader, _ = load_dataset(batch_size=batchSize, transform=augmentedTransform)
    display_sample(original_loader=originalTestloader,
                   augmented_loader=augmentedTestloader)