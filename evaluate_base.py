import torch
from torchvision import transforms
from utils.dataset import load_dataset

import wrn as wideresnet

# Load the saved model weights
net = wideresnet.WideResNet_28_4(10, 'CIFAR10', normalized=True, block=wideresnet.WideBasic, activation_function='silu')
PATH = '/home/ekagra/Documents/GitHub/MasterArbeit/models/robust.pth'

net = torch.nn.DataParallel(net)
net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu'))['model_state_dict'], strict=False)

net.eval()

# Prepare the DataLoader
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
])
trainloader, testloader, _ = load_dataset(batch_size=100, transform=transform)

# Evaluate the model
correct, total = 0, 0
with torch.no_grad():
    net.eval()
    for i, data in enumerate(testloader):
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        accuracy = correct / total
        if (i + 1) % 50 == 0:
            print(
                f"Processed [{i+1}/{len(testloader)}] - Accuracy: {accuracy*100:.2f}%"
            )
    print(
        f"Accuracy of the network on the CIFAR-10 test dataset: {accuracy * 100:.2f} %"
    )
