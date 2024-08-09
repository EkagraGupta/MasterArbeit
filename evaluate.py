import torch

import wrn as wideresnet
from augment_dataset import create_transforms, load_data
from compute_loss import soft_loss


def evaluate_model(model, dataloader):
    # Evaluate the model
    correct, total = 0, 0

    with torch.no_grad():
        model.eval()
        for _, data in enumerate(dataloader):
            images, labels, _ = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = correct / total
        # print(
        #     f"Accuracy of the network on the CIFAR-10 test dataset: {accuracy * 100:.2f} %"
        # )
        return accuracy


if __name__ == "__main__":
    # Load the saved model weights
    # net_path = "/home/ekagra/Documents/GitHub/MasterArbeit/models/cifar_net_baseline.pth"
    # net = WideResNet_28_4(num_classes=10)
    # net.load_state_dict(torch.load(net_path, map_location=torch.device("cpu")))

    # Load the saved model weights
    net = wideresnet.WideResNet_28_4(10, 'CIFAR10', normalized=True, block=wideresnet.WideBasic, activation_function='silu')
    state_dict_key = "model_state_dict"
    PATH = '/home/ekagra/Documents/GitHub/MasterArbeit/models/robust.pth'

    net = torch.nn.DataParallel(net)
    state_dict = torch.load(PATH, map_location=torch.device('cpu'))
    net.load_state_dict(state_dict[state_dict_key], strict=False)

    # net.eval()


    # Prepare the DataLoader
    transforms_preprocess, transforms_augmentation = create_transforms(
        random_cropping=False,
        aggressive_augmentation=True,
        custom=False,
        augmentation_name="Brightness",
        augmentation_severity=30,
    )
    custom_trainset, custom_testset = load_data(
        transforms_augmentation=transforms_augmentation,
        transforms_preprocess=transforms_preprocess,
        dataset_split=1000,
    )

    custom_dataloader = torch.utils.data.DataLoader(
        custom_testset, batch_size=1, shuffle=False
    )
    accuracy = evaluate_model(model=net, dataloader=custom_dataloader)
    print(f"Accuracy: {accuracy*100:.2f}%")
