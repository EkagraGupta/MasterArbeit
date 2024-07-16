from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from utils.trivial_augment import TrivialAugmentWide

from dump.dataset import load_dataset
from utils.ncc import normalized_cross_correlation
from utils.sift_comparison import sift_correction_factor
import piq


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
        pixelwise_augs = [
            "Invert",
            "Equalize",
            "AutoContrast",
            "Posterize",
            "Solarize",
            "SolarizeAdd",
            "Color",
            "Contrast",
            "Brightness",
            "Sharpness",
        ]
        trivial_augment = TrivialAugmentWide()
        augmented_image, image_info = trivial_augment(image)
        augmentation_type = next(iter(image_info.keys()))
        # print(f"\nInitial tr: {image_info[augmentation_type]}")
        if augmentation_type in pixelwise_augs:
            resize = transforms.Resize((41, 41))
            # Apply the resize transformation to both the original and augmented images
            to_pil = transforms.ToPILImage()
            to_tensor = transforms.ToTensor()
            im_resize = to_tensor(resize(to_pil(image)))
            augmented_im_resize = to_tensor(resize(to_pil(augmented_image)))
            vif_value = piq.vif_p(
                im_resize.unsqueeze(0), augmented_im_resize.unsqueeze(0)
            )
            # image_info[augmentation_type] = vif_value.item()
            confidence_aa = vif_value.item()
        else:
            # image_info[augmentation_type] = sift_correction_factor(
            #     original_image=image, augmented_image=augmented_image
            # )
            confidence_aa = sift_correction_factor(
                original_image=image, augmented_image=augmented_image
            )
        print(f"\nAugmentation info: {image_info}\n")
        # return augmented_image, image_info
        return augmented_image, confidence_aa


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize((512, 512)), transforms.ToTensor()]
    )
    trainloader, _, classes = load_dataset(batch_size=10, transform=transform)

    images, labels = next(iter(trainloader))
    ta = CustomTrivialAugmentWide()
    new_image, aug_info = ta(images[0])
    print(aug_info)

    new_image = new_image.squeeze(0)
    new_image_np = new_image.permute(1, 2, 0).numpy()
    plt.imsave(
        "/home/ekagra/Desktop/Study/MA/code/example/augmented_example_image.png",
        new_image_np,
    )
    plt.imshow(new_image_np)
    plt.title("Augmented Image")
    # plt.show()
