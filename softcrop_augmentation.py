import numpy as np

import torch
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as ff

from dataset import load_dataset


class SoftCropAugmentation:
    def __init__(self, n_class=10, sigma=0.3, k=2, sigma_crop=10):
        self.chance = 1 / n_class
        self.sigma = sigma
        self.k = k
        self.sigma_crop = sigma_crop

    def draw_offset(self, limit, sigma=0.3, n=100):
        # Draw an integer from a (clipped) gaussian

        for d in range(n):
            x = torch.randn((1)) * sigma
            if abs(x) <= limit:
                return int(x)
        return int(0)

    def __call__(self, image, label):
        # Extract image dimensions
        dim1, dim2 = image.size(1), image.size(2)

        # Pad image
        image_padded = torch.zeros((3, dim1 * 3, dim2 * 3))

        # draw tx, ty
        tx = self.draw_offset(dim1, self.sigma_crop * dim1)
        ty = self.draw_offset(dim2, self.sigma_crop * dim2)

        # crop image
        left, right = tx + dim1, tx + dim2 * 2
        top, bottom = ty + dim2, ty + dim2 * 2
        new_image = image_padded[:, left:right, top:bottom]

        # compute transformed image visibility and confidence
        v = (dim1 - abs(tx)) * (dim2 - abs(ty)) / (dim1 * dim2)
        confidence = 1 - (1 - self.chance) * (1 - v) ** self.k
        return new_image, label, confidence


if __name__ == "__main__":
    softcrop_augment = SoftCropAugmentation()
    outputs = torch.tensor(
        [
            [
                0.01,  # 0: airplane
                0.01,  # 1: automobile
                0.01,  # 2: bird
                0.01,  # 3: cat
                0.01,  # 4: deer
                0.01,  # 5: dog
                0.91,  # 6: frog
                0.01,  # 7: horse
                0.01,  # 8: ship
                0.01,  # 9: truck
            ],
        ]
    )

    reweight = True
    soften_one_hot = True

    transform = transforms.Compose([transforms.ToTensor()])
    trainloader, _, classes = load_dataset(batch_size=1, transform=transform)

    images, labels = next(iter(trainloader))
    print(f"\nOriginal Hard label: {labels} -> {classes[labels.item()]}\n")
    # cropped_image, new_label = soft_crop(images[0], labels)
    # soft_one_hot, loss = soft_target(
    #     pred=outputs, gold=new_label, reweight=reweight, soften_one_hot=soften_one_hot
    # )
    new_image, label, confidence = softcrop_augment(images[0], labels)

    pil_image = ff.to_pil_image(new_image)
