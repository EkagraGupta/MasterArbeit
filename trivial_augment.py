from utils.augment_parameters import get_augmentation_info
from torchvision import transforms
from dump.dataset import load_dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class TrivialAug:
    def __init__(
            self
    ):
        pass

    def __call__(self, image):
        augmented_image, augment_info = get_augmentation_info(image)
        return augmented_image, augment_info
    

if __name__=='__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    trainloader, _, classes = load_dataset(batch_size=1, transform=transform)
    
    images, labels = next(iter(trainloader))

    ta = TrivialAug()
    new_image, aug_info = ta(images[0])
    print(new_image.shape)
    print(aug_info)
    # Remove the extra batch dimension
    new_image = new_image.squeeze(0)

    # If the image is in shape (3, 32, 32), it needs to be transposed to (32, 32, 3) for matplotlib
    new_image_np = new_image.permute(1, 2, 0).numpy()

    # Display the image
    plt.imshow(new_image_np)
    plt.title('Augmented Image')
    plt.show()
