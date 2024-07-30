from PIL import Image
import numpy as np
from torchvision import transforms
from dataset import load_dataset

transform = transforms.Compose([transforms.Resize(512),
                                transforms.RandomRotation(40),
                                transforms.ToTensor()])
to_pil = transforms.ToPILImage()
trainloader, testloader, classes = load_dataset(batch_size=1, transform=transform)

images, labels = next(iter(trainloader))
im = to_pil(images[0])
im.save('/home/ekagra/Documents/GitHub/MasterArbeit/example/augmented_image.png')