import torch
import torchvision
from PIL import Image
from torchvision import transforms, datasets
from augment_dataset import create_transforms, load_data

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor()
])

batch_size = 1
DATASET_NAME = "CIFAR10"
augmentation_type = "TranslateX"
severity = 30
augment_sign = True

g = torch.Generator()
g.manual_seed(1)

transforms_preprocess, transforms_augmentation = create_transforms(
    random_cropping=False,
    aggressive_augmentation=True,
    custom=True,
    augmentation_name=augmentation_type,
    augmentation_severity=severity,
    augmentation_sign=augment_sign,
    dataset_name=DATASET_NAME
)

print(f"Preprocess transforms: {transforms_preprocess}\nAugmentation transforms: {transforms_augmentation}")

trainset, testset = load_data(
    transforms_preprocess=transforms_preprocess,
    transforms_augmentation=transforms_augmentation,
    dataset_name=DATASET_NAME
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=False)
classes = trainset.dataset.classes
images, labels, confidences = next(iter(trainloader))

to_pil = transforms.ToPILImage()
im = to_pil(images[0])
# im.show()
if augment_sign:
    im.save(f"final_plots/augmented_{augmentation_type}_min_{severity}.png")
else:
    im.save(f"final_plots/augmented_{augmentation_type}_plus_{severity}.png")
