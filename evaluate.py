import torch

from wideresnet import WideResNet_28_4
from load_augmented_dataset import get_dataloader

# Load the saved model weights
net_path = "/home/ekagra/Desktop/Study/MA/code/models/cifar_net_da0_aa1.pth"
net = WideResNet_28_4(num_classes=10)
net.load_state_dict(torch.load(net_path, map_location=torch.device("cpu")))
net.eval()  # set the model to evaluation mode

# Prepare the DataLoader
custom_dataloader = get_dataloader(
    num_samples=None, train=False, da=2, aa=1, normalize=False
)

# Evaluate the model
correct, total = 0, 0

with torch.no_grad():
    net.eval()
    for data in custom_dataloader:
        images, labels, confidence = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(
        f"Accuracy of the network on the CIFAR-10 test dataset: {accuracy * 100:.2f} %"
    )
