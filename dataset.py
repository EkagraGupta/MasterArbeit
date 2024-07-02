import torch
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
from PIL import Image


def load_dataset(batch_size: int, transform):
    trainset = datasets.CIFAR10(
        root="./data/train", train=True, download=True, transform=transform
    )
    testset = datasets.CIFAR10(
        root="./data/test", train=False, download=True, transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=False
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    classes = testset.classes
    return trainloader, testloader, classes


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])

    _, testloader, classes = load_dataset(batch_size=1, transform=transform)
    print(classes)
    for i, (images, labels) in enumerate(testloader):
        print(labels)
        pil_image = F.to_pil_image(images[0])
        image_path = "/home/ekagra/Desktop/Study/MA/code/example/example_image.png"
        pil_image.save(image_path)
        break
