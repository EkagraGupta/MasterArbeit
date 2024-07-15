from torchvision import datasets, transforms
from dump.dataset import load_dataset
from PIL import Image
import numpy as np

transform = transforms.Compose(
    [
        # transforms.RandomAffine(80),
        transforms.Resize((512, 512)),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
    ]
)
trainloader, _, classes = load_dataset(batch_size=1, transform=transform)

images, labels = next(iter(trainloader))

im_numpy = images[0].squeeze().numpy()
im_numpy = np.transpose(im_numpy, (1, 2, 0))
im_numpy = (im_numpy * 255).astype(np.uint8)
pil_im = Image.fromarray(im_numpy)
pil_im.save("/home/ekagra/Desktop/Study/MA/code/example/augmented_example_image.png")
pil_im.show()
