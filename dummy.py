from augment_dataset import create_transforms, load_data
import torch
from utils.measure_dataloader import measure_dataloader_time
from torchvision import transforms, datasets
from torchvision.transforms import functional as F

preprocess, augmentation = create_transforms(
    random_cropping=False,
    aggressive_augmentation=True,
    custom=True,
    augmentation_name="Sharpness",
    augmentation_severity=30,
    augmentation_sign=True,
)

trainset, _ = load_data(
    transforms_preprocess=preprocess,
    transforms_augmentation=augmentation,
    dataset_split=10,
)
# magnitude = 0.99
# sharpness_factor = 1.0 + magnitude
# adjust_sharpness = transforms.Lambda(lambda x: F.adjust_sharpness(x, sharpness_factor))

# transform_train = transforms.Compose(
#     [   
#         transforms.GaussianBlur(kernel_size=5, sigma=1.0),
#         transforms.ToTensor(),
#     ]
# )

# trainset = datasets.CIFAR10(
#     root="./data", train=True, download=True, transform=transform_train
# )

trainloader = torch.utils.data.DataLoader(trainset, shuffle=False, batch_size=1)

# time_taken = measure_dataloader_time(trainloader)
# print(f"\nTime taken: {time_taken:.3f} seconds.\n")

images, labels, conf = next(
    iter(trainloader)
)  # confidences[0]: augmentation_magnitude, confidences[1]: confidence_score
to_pil = transforms.ToPILImage()
resize = transforms.Resize((256, 256))
im_pil = resize(to_pil(images[0]))
im_pil.show()

# Gaussian blur image
# from PIL import Image, ImageFilter
# im = Image.open('example/original_image.png')
# im = im.filter(ImageFilter.GaussianBlur(radius=1.0))
# im = im.resize((256, 256))
# im.show()
