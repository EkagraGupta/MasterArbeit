import torch

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
# custom_dataloader = get_dataloader(num_samples=None, train=False, da=2, aa=1)
transforms_preprocess, transforms_augmentation = create_transforms(
    random_cropping=True, aggressive_augmentation=True, custom=True
)
custom_trainset, custom_testset = load_data(
    transforms_augmentation=transforms_augmentation,
    transforms_preprocess=transforms_preprocess,
)

custom_dataloader = torch.utils.data.DataLoader(custom_testset, batch_size=1)

# Evaluate the model
correct, total = 0, 0

with torch.no_grad():
    net.eval()
    for i, data in enumerate(custom_dataloader):
        images, labels, confidences = data
        if len(images) > 1:
            confidences = images[1]
            images = images[0]
        # print(confidences)
        # break
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
