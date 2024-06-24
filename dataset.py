import torch
from torchvision import datasets, transforms


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

    _, testloader, _ = load_dataset(batch_size=1, transform=transform)

    for i, (images, labels) in enumerate(testloader):
        print(labels)
        break
