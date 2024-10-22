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
            self.root_dir = os.path.join(root, 'train')
            self.wnids = self.load_wnids(os.path.join(root, 'wnids.txt'))
            self.class_names = self.load_class_names(os.path.join(root, 'words.txt'))
            self.class_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}
            
            for class_name, idx in self.class_to_idx.items():
                class_dir = os.path.join(self.root_dir, class_name)
                class_dir = os.path.join(class_dir, 'images')
                for image_name in os.listdir(class_dir):
                    im_path = os.path.join(class_dir, image_name)
                    if image_name.endswith('.JPEG'):
                        self.data.append(im_path)
                        self.labels.append(idx)
        else:
            self.root_dir = os.path.join(root, 'val')
            self.wnids = self.load_class_names(os.path.join(self.root_dir, 'val_annotations.txt'))
            self.class_names = self.load_class_names(os.path.join(root, 'words.txt'))
            self.wnids_data = {}

            for im_name, wnid_data in self.wnids.items():
                wnid = wnid_data.split(' ')[0]
                self.wnids_data[im_name] = wnid
            
            self.idx_to_wnids = {idx: wnid for idx, wnid in enumerate(self.wnids_data.values())}

            for idx, (im_name, wnid) in enumerate(self.wnids_data.items()):
                im_path = os.path.join(self.root_dir, 'images', im_name)
                self.data.append(im_path)
                self.labels.append(idx)
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
        with open(wnids_path, 'r') as f:
            wnids = [line.strip() for line in f.readlines()]
        return wnids

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
        if self.train:
            wnid = self.wnids[label]
            return self.class_names.get(wnid, 'Unknown')
        else:
            wnid = self.idx_to_wnids[label]
            return self.class_names.get(wnid, 'Unknown')
    
def display_image_grid(images, labels, batch_size):
    grid_im = torchvision.utils.make_grid(images, nrow=batch_size//10)
    np_im = grid_im.numpy()

    plt.figure(figsize=(batch_size * 2, 2))
    plt.imshow(np.transpose(np_im, (1, 2, 0)))
    plt.axis('off')

    for i in range(batch_size):
        ax = plt.subplot(1, batch_size, i + 1)
        ax.imshow(np.transpose(images[i].numpy(), (1, 2, 0)))
        ax.set_title(f"{labels[i].item()}: {dataset.get_class_name(label[i].item())}")
        ax.axis('off')
    plt.subplots_adjust(wspace=1, hspace=0.5)
    plt.show()

if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    dataset = TinyImageNetDataset('/home/ekagra/Documents/GitHub/MasterArbeit/data/tiny_imnet/tiny-imagenet-200', transform=transform, train=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    image, label = next(iter(dataloader))
    display_image_grid(image, label, 10)
    