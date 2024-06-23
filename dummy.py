from torchvision import transforms

from dataset import load_dataset
from utils.augment_parameters import get_augmentation_info

basic_transform = transforms.Compose([transforms.ToTensor()])

train_loader, _, _ = load_dataset(batch_size=1, transform=basic_transform)
for i, (images, labels) in enumerate(train_loader):
    if i > 0:
        break
    augmented_image, augmentation_info = get_augmentation_info(images)
    print(augmentation_info)
