from augment_dataset import create_transforms, load_data
import torch
from utils.measure_dataloader import measure_dataloader_time
from torchvision import transforms

preprocess, augmentation = create_transforms(
    random_cropping=False,
    aggressive_augmentation=True,
    custom=True,
    augmentation_name="Brightness",
    augmentation_severity=20,
    augmentation_sign=False,
)

trainset, _ = load_data(
    transforms_preprocess=preprocess,
    transforms_augmentation=augmentation,
    dataset_split=10,
)

trainloader = torch.utils.data.DataLoader(trainset, shuffle=False, batch_size=1)

# time_taken = measure_dataloader_time(trainloader)
# print(f"\nTime taken: {time_taken:.3f} seconds.\n")

images, labels, confidences = next(
    iter(trainloader)
)  # confidences[0]: augmentation_magnitude, confidences[1]: confidence_score
to_pil = transforms.ToPILImage()
resize = transforms.Resize((256, 256))
im_pil = resize(to_pil(images[0]))
im_pil.show()
