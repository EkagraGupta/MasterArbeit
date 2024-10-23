from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class TinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, train=True):
        self.train = train
        self.data = []
        self.labels = []

        if train:
            # Training data
            self.root_dir = os.path.join(root, 'train')
            self.wnids = self.load_wnids(os.path.join(root, 'wnids.txt'))
            self.class_names = self.load_class_names(os.path.join(root, 'words.txt'))
            self.class_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}
            
            for class_name, idx in self.class_to_idx.items():
                class_dir = os.path.join(self.root_dir, class_name, 'images')
                for image_name in os.listdir(class_dir):
                    if image_name.endswith('.JPEG'):
                        im_path = os.path.join(class_dir, image_name)
                        self.data.append(im_path)
                        self.labels.append(idx)
        else:
            # Validation data
            self.root_dir = os.path.join(root, 'val')
            self.val_annotations = self.load_val_annotations(os.path.join(self.root_dir, 'val_annotations.txt'))
            self.wnids = self.load_wnids(os.path.join(root, 'wnids.txt'))  # WNIDs from training
            self.class_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}  # Map WNIDs to class indices

            for im_name, wnid in self.val_annotations.items():
                im_path = os.path.join(self.root_dir, 'images', im_name)
                self.data.append(im_path)
                if wnid in self.class_to_idx:
                    self.labels.append(self.class_to_idx[wnid])  # Correct label mapping to class idx
                else:
                    raise ValueError(f"WNID {wnid} not found in class_to_idx mapping.")

        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, label

    def load_wnids(self, wnids_path):
        # Load WNIDs from wnids.txt (same for training and validation)
        with open(wnids_path, 'r') as f:
            wnids = [line.strip() for line in f.readlines()]
        return wnids

    def load_val_annotations(self, val_annotations_path):
        # Load validation annotations from val_annotations.txt
        val_annotations = {}
        with open(val_annotations_path, 'r') as f:
            for line in f:
                parts = line.split('\t')
                image_name = parts[0]
                wnid = parts[1]
                val_annotations[image_name] = wnid  # Map: image filename -> WNID
        return val_annotations


    def load_class_names(self, words_path):
        class_names = {}
        with open(words_path, 'r') as f:
            for line in f:
                parts = line.split('\t')
                wnid = parts[0]
                name = ' '.join(parts[1:])
                class_names[wnid] = name
        return class_names

    def get_class_name(self, label):
        wnid = self.wnids[label] if self.train else list(self.class_to_idx.keys())[label]
        return self.class_names.get(wnid, 'Unknown')
    
def display_image_grid(images, labels, batch_size, dataset):
    grid_im = torchvision.utils.make_grid(images, nrow=batch_size // 10)
    np_im = grid_im.numpy()

    plt.figure(figsize=(batch_size * 2, 2))
    plt.imshow(np.transpose(np_im, (1, 2, 0)))
    plt.axis('off')

    for i in range(batch_size):
        ax = plt.subplot(1, batch_size, i + 1)
        ax.imshow(np.transpose(images[i].cpu().numpy(), (1, 2, 0)))
        ax.set_title(f"{labels[i].item()}: {dataset.get_class_name(labels[i].item())}")
        ax.axis('off')
    
    plt.subplots_adjust(wspace=1, hspace=0.5)
    plt.show()

def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            print(f'Predicted: {predicted}\tLabels: {labels}')
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total
    print(f'Validation accuracy: {val_acc:.2f}')

    

if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Add normalization for ImageNet
    ])

    trainset = TinyImageNetDataset('/home/ekagra/Documents/GitHub/MasterArbeit/data/tiny_imnet/tiny-imagenet-200', transform=transform, train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)
    valset = TinyImageNetDataset('/home/ekagra/Documents/GitHub/MasterArbeit/data/tiny_imnet/tiny-imagenet-200', transform=transform, train=False)
    valloader = torch.utils.data.DataLoader(valset, batch_size=10, shuffle=False)

    # display_image_grid(*next(iter(valloader)), 10, valset)

    model = torchvision.models.resnet18(pretrained=True)

    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 200)  # TinyImageNet has 200 classes

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    evaluate(model, trainloader)
