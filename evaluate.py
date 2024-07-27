import torch

from wideresnet import WideResNet_28_4
from augment_dataset import create_transforms, load_data
from compute_loss import soft_loss

# Load the saved model weights
# net_path = "/home/ekagra/Desktop/Study/MA/code/models/cifar_net_da0_aa1.pth"
# net_path = "/home/ekagra/Desktop/Study/MA/code/models/cifar_net.pth"
net_path = '/home/ekagra/Documents/GitHub/MasterArbeit/models/cifar_net_exp01.pth'
net = WideResNet_28_4(num_classes=10)
net.load_state_dict(torch.load(net_path, map_location=torch.device("cpu")))
net.eval()  # set the model to evaluation mode

# Prepare the DataLoader
transforms_preprocess, transforms_augmentation = create_transforms(
    random_cropping=True, aggressive_augmentation=True, custom=False
)
custom_trainset, custom_testset = load_data(
    transforms_augmentation=transforms_augmentation,
    transforms_preprocess=transforms_preprocess,
)

custom_dataloader = torch.utils.data.DataLoader(
    custom_testset, batch_size=128, shuffle=False
)

# Evaluate the model
correct, total = 0, 0

with torch.no_grad():
    net.eval()
    for i, data in enumerate(custom_dataloader):
        images, labels, confidences = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        accuracy = correct / total
        if (i + 1) % 1000 == 0:
            print(
                f"Processed [{i+1}/{len(custom_dataloader)}] - Accuracy: {accuracy*100:.2f}%"
            )
    print(
        f"Accuracy of the network on the CIFAR-10 test dataset: {accuracy * 100:.2f} %"
    )
