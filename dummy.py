from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt

class TinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, train=True):
        self.train = train
        if train:
            self.root_dir = os.path.join(root, 'train')
        else:
            self.root_dir = os.path.join(root, 'val')
        self.transform = transform
        self.data = []
        self.labels = []
        self.wnids = self.load_wnids(os.path.join(root, 'wnids.txt'))
        self.class_names = self.load_class_names(os.path.join(root, 'words.txt'))

        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}

        for class_name, idx in self.class_to_idx.items():
            if train:
                class_dir = os.path.join(self.root_dir, class_name)
            else:
                class_dir = self.root_dir
            class_dir = os.path.join(class_dir, 'images')
            for image_name in os.listdir(class_dir):
                im_path = os.path.join(class_dir, image_name)
                if image_name.endswith('.JPEG'):
                    self.data.append(im_path)
                    self.labels.append(idx)

    
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
        wnid = self.wnids[label]
        return self.class_names.get(wnid, 'Unknown')
    

if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    dataset = TinyImageNetDataset('/home/ekagra/Documents/GitHub/MasterArbeit/data/tiny_imnet/tiny-imagenet-200', transform=transform, train=True)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    image, label = next(iter(dataloader))
    plt.imshow(image[0].permute(1, 2, 0))
    plt.title(dataset.get_class_name(label[0].item()))
    plt.show()
    