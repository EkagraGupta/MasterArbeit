import torch
from torchvision import transforms
from utils.dataset import load_dataset

from wideresnet import WideResNet_28_4
from dump.load_augmented_dataset import get_dataloader
from augment_dataset import create_transforms, load_data

# Load the saved model weights
net_path = "/home/ekagra/Desktop/Study/MA/code/models/cifar_net_da0_aa1.pth"
# net_path = "/home/ekagra/Desktop/Study/MA/code/models/cifar_net.pth"
net = WideResNet_28_4(num_classes=10)
net.load_state_dict(torch.load(net_path, map_location=torch.device("cpu")))
net.eval()  # set the model to evaluation mode

# Prepare the DataLoader
transform = transforms.Compose([transforms.ToTensor()])
trainloader, testloader, _ = load_dataset(batch_size=128, transform=transform)

# Evaluate the model
correct, total = 0, 0

with torch.no_grad():
    net.eval()
    for i, data in enumerate(testloader):
        images, labels = data
        # print(images.shape)
        # if len(images) > 1:
        #     confidences = images[1]
        #     images = images[0]
        # print(confidences)
        # break
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        accuracy = correct / total
        if (i+1)%1000==0:
            print(f'Processed [{i+1}/{len(testloader)}] - Accuracy: {accuracy*100:.2f}%')
    print(
        f"Accuracy of the network on the CIFAR-10 test dataset: {accuracy * 100:.2f} %"
    )
