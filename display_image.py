from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision

from dataset import load_dataset

def display_sample(dataloader):

    for i, (image, label) in enumerate(dataloader):
        image = torchvision.utils.make_grid(image)
        imageNumpy = image.numpy()
        plt.imshow(np.transpose(imageNumpy, (1, 2, 0)))
        plt.show()

        if i==5:
            break

if __name__ == '__main__':
    trainloader, testloader, classes = load_dataset(batch_size=1)
    display_sample(dataloader=testloader)