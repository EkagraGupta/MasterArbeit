import torch
from torchvision import datasets, transforms
from utils.dataset import load_dataset

transform = transforms.Compose(
    [   
        transforms.Resize(512),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        # transforms.TrivialAugmentWide(),
    ]
)

trainloader, testloader, classes = load_dataset(batch_size=1, transform=transform)

images, labels = next(iter(trainloader))

to_pil = transforms.ToPILImage()
pil_im = to_pil(images[0])
pil_im.save('/home/ekagra/Documents/GitHub/MasterArbeit/example/augmented_image.png')