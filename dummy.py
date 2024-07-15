from torchvision import datasets, transforms
from dump.dataset import load_dataset
from PIL import Image
import numpy as np

transform = transforms.Compose(
    [
        # transforms.RandomAffine(80),
        transforms.Resize((512, 512)),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ]
)
# trainloader, _, classes = load_dataset(batch_size=1, transform=transform)

# images, labels = next(iter(trainloader))
image_path = "/home/ekagra/Desktop/Study/MA/code/example/resized_example_image.png"
image = Image.open(image_path)

# image_np = images[0].squeeze().numpy()
# image_np = np.transpose(image_np, (1, 2, 0))
# image_np = (image_np * 255).astype(np.uint8)
new_image = transform(image)
new_image = new_image.numpy()
new_image = (new_image * 255).astype(np.uint8)
new_image = np.transpose(new_image, (1, 2, 0))
pil_image = Image.fromarray(new_image)
pil_image.save("/home/ekagra/Desktop/Study/MA/code/example/augmented_example_image.png")
pil_image.show()
