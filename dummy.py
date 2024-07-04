from torchvision import datasets, transforms
from dump.dataset import load_dataset
from PIL import Image
import numpy as np

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.TrivialAugmentWide(),
    ]
)
trainloader, _, classes = load_dataset(batch_size=1, transform=transform)

images, labels = next(iter(trainloader))

image_np = images[0].squeeze().numpy()
image_np = np.transpose(image_np, (1, 2, 0))
image_np = (image_np * 255).astype(np.uint8)
pil_image = Image.fromarray(image_np)
pil_image.save("/home/ekagra/Desktop/Study/MA/code/example/augmented_image.png")
pil_image.show()