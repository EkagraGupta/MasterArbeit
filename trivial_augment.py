import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from utils.trivial_augment import TrivialAugmentWide

from dump.dataset import load_dataset


class CustomTrivialAugmentWide:
    """A class to perform trivial augmentation on images which
    returns augmented image and its augmentation information.
    """

    def __init__(self):
        """Initializes the CustomTrivialAugmentWide class."""
        pass

    def __call__(self, image):
        """Applies the trivial augmentation to input images.

        Args:
            image (torch.Tensor): The input image tensor.

        Returns:
            tuple: The augmented image tensor and augmentation information.
        """
        augmented_image, augment_info = self.get_augment_info(image)
        return augmented_image, augment_info

    @staticmethod
    def get_augment_info(image):
        """Applies trivial augmentaiton to the input image.

        Args:
            image (torch.Tensor): The input image tensor.

        Returns:
            tuple: The augmented image tensor and augmentation information.
        """
        trivial_augment = TrivialAugmentWide()
        augmented_pil_image, image_info = trivial_augment(image)
        # print(type(augmented_pil_image))
        return augmented_pil_image, image_info


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    trainloader, _, classes = load_dataset(batch_size=1, transform=transform)

    images, labels = next(iter(trainloader))

    ta = CustomTrivialAugmentWide()
    new_image, aug_info = ta(images[0])
    print(aug_info)

    # Remove the extra batch dimension
    new_image = new_image.squeeze(0)

    # If the image is in shape (3, 32, 32), it needs to be transposed to (32, 32, 3) for matplotlib
    new_image_np = new_image.permute(1, 2, 0).numpy()

    # Display the image
    plt.imshow(new_image_np)
    plt.title("Augmented Image")
    plt.show()
