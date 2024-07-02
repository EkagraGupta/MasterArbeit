from torchvision import transforms

from dump.dataset import load_dataset
from utils.augment_parameters import get_augmentation_info
from utils.correction_function import get_correction_factor
from utils.softlabel_tensor import get_softlabel_tensor
from utils.display_image import get_images

basic_transform = transforms.Compose(
    [transforms.Resize((100, 100)), transforms.ToTensor()]
)

train_loader, _, classes = load_dataset(batch_size=1, transform=basic_transform)
for i, (images, labels) in enumerate(train_loader):
    if i > 0:
        break
    augmented_image, augmentation_info = get_augmentation_info(images)
    correction_factor = get_correction_factor(augmentation_info)
    label_dict, label_tensor = get_softlabel_tensor(labels.item(), correction_factor)
    # get_images(images, augmented_image)
