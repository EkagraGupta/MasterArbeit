import torch
from torchvision import transforms
from utils.dataset import load_dataset

# import wrn as wideresnet
from wideresnet import WideResNet_28_4, WideBasic

# Load the saved model weights
net = WideResNet_28_4(num_classes=10)
PATH = '/home/ekagra/Documents/GitHub/MasterArbeit/models/robust_no_TA_augments.pth'
net = torch.nn.DataParallel(net)
state_dict = torch.load(PATH, map_location=torch.device('cpu'))
net.load_state_dict(state_dict["model_state_dict"], strict=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# net = WideResNet_28_4(num_classes=10)  # Initialize the same mode/l class
# net_path = '/home/ekagra/Documents/GitHub/MasterArbeit/models/robust_no_TA_augments.pth'
# net = torch.nn.DataParallel(net)
# state_dict = torch.load(net_path, map_location=torch.device('cpu'))
# # print(state_dict.keys())
# net.load_state_dict(state_dict, strict=False)
# # net.to(device)
net.to(device)

# Prepare the DataLoader
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    # transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
])
trainloader, testloader, _ = load_dataset(batch_size=1000, transform=transform)

# Evaluate the model
correct, total = 0, 0
with torch.no_grad():
    net.eval()
    for i, data in enumerate(testloader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
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
