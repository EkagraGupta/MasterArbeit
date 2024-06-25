import numpy as np

import torch
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as ff

from dataset import load_dataset


def soft_target(
    pred,
    gold,
    other=False,
    distribute=True,
    reweight=False,
    soften_one_hot=True,
    lr_correction=False,
):
    # print(gold)
    gold = gold.unsqueeze(1)
    target = gold.long()
    prob = 1 - (gold - target)
    # print()
    weight = torch.clone(prob).float() if reweight else torch.ones_like(prob).float()
    if lr_correction:
        weight = weight / weight.mean()
    n_class = pred.size(1)
    # if we distribute 1-prob to other classes
    scatter_mul = 1.0 if distribute else 0.0
    if soften_one_hot:
        if not other:  # if there is an other class
            one_hot = (
                torch.ones_like(pred) * (1 - prob) * scatter_mul / (n_class - 1)
            ).float()
            one_hot.scatter_(dim=1, index=target, src=prob.float())
        else:
            one_hot = torch.zeros_like(pred)
            one_hot.scatter_(
                dim=1,
                index=torch.ones_like(target) * (n_class - 1),
                src=(1 - prob.float()) * scatter_mul,
            )
            one_hot.scatter_(dim=1, index=target, src=prob.float())
    else:
        one_hot = torch.zeros_like(pred)
        one_hot.scatter_(dim=1, index=target, src=torch.ones_like(target).float())

    log_prob = F.log_softmax(pred, dim=1)

    kl = weight * F.kl_div(
        input=log_prob.float(), target=one_hot.float(), reduction="none"
    ).sum(-1)

    return one_hot, kl.mean()


def ComputeProb(x, T=0.25, n_classes=10, max_prob=1.0, pow=2.0):
    max_prob = torch.clamp_min(torch.tensor(max_prob), 1 / n_classes)
    if T <= 0:
        T = 1e-10

    if x > T:
        return max_prob
    elif x > 0:
        a = (max_prob - 1 / float(n_classes)) / (T**pow)
        return max_prob - a * (T - x) ** pow
    else:
        return np.ones_like(x) * 1 / n_classes


class SoftCrop:
    """
    crop image

    """

    def __init__(
        self,
        n_class=10,
        sigma_crop=10,
        t_crop=1.0,
        max_p_crop=1.0,
        pow_crop=4.0,
        bg_crop=0.01,
        iou=False,
    ):
        self.n_class = n_class
        self.chance = 1 / n_class

        # crop parameters
        self.sigma_crop = sigma_crop
        self.t_crop = t_crop
        self.max_p_crop = max_p_crop
        self.pow_crop = pow_crop
        self.bg_crop = bg_crop

        self.iou = iou  # if true, use IoU to compute r, else use IoForeground
        # for debugging
        self.flag = True

        # print("use soft crop")
        # print(
        #     "sigma: ",
        #     self.sigma_crop,
        #     " T: ",
        #     self.t_crop,
        #     " Max P: ",
        #     self.max_p_crop,
        #     "bg: ",
        #     self.bg_crop,
        #     "power: ",
        #     self.pow_crop,
        #     "IoU: ",
        #     self.iou,
        # )

    def draw_offset(self, sigma=1, limit=24, n=100):
        # draw an integer from gaussian within +/- limit
        for d in range(n):
            x = torch.randn((1)) * sigma
            if abs(x) <= limit:
                return int(x)
        return int(0)

    def __call__(self, image, label):
        dim1 = image.size(1)
        dim2 = image.size(2)

        # Soft Crop
        # bg = torch.randn((3,dim1*3,dim2*3)) * self.bg_crop # create a 3x by 3x sized noise background
        bg = (
            torch.ones((3, dim1 * 3, dim2 * 3)) * self.bg_crop * torch.randn((3, 1, 1))
        )  # create a 3x by 3x sized noise background
        bg[:, dim1 : 2 * dim1, dim2 : 2 * dim2] = image  # put image at the center patch
        pil_bg = ff.to_pil_image(bg)
        pil_bg.save("/home/ekagra/Desktop/Study/MA/code/example/example_bg_image.png")

        offset1 = self.draw_offset(self.sigma_crop, dim1)
        offset2 = self.draw_offset(self.sigma_crop, dim2)

        left = offset1 + dim1
        top = offset2 + dim2
        right = offset1 + dim1 * 2
        bottom = offset2 + dim2 * 2

        # number of pixels in orignal image kept after cropping alone
        intersection = (dim1 - abs(offset1)) * (dim2 - abs(offset2))
        # proportion of original pixels left after cutout and cropping
        if self.iou:
            overlap = intersection / (dim1 * dim2 * 2 - intersection)
        else:
            overlap = intersection / (dim1 * dim2)
        # now the max prob can not be larger than prob_mix
        prob_crop = ComputeProb(
            overlap,
            T=self.t_crop,
            max_prob=self.max_p_crop,
            pow=self.pow_crop,
            n_classes=self.n_class,
        )

        new_image = bg[:, left:right, top:bottom]  # crop image
        new_label = label + 1 - prob_crop  # max(prob_crop*prob_mix,self.chance)
        return new_image.clone().detach(), new_label.clone().detach()


if __name__ == "__main__":
    outputs = torch.tensor(
        [
            [
                0.01,  # 0: airplane
                0.01,  # 1: automobile
                0.01,  # 2: bird
                0.01,  # 3: cat
                0.01,  # 4: deer
                0.01,  # 5: dog
                0.99,  # 6: frog
                0.01,  # 7: horse
                0.01,  # 8: ship
                0.01,  # 9: truck
            ],
        ]
    )
    reweight = True
    soften_one_hot = True

    soft_crop = SoftCrop()
    transform = transforms.Compose([transforms.ToTensor()])
    trainloader, _, classes = load_dataset(batch_size=1, transform=transform)

    images, labels = next(iter(trainloader))
    print(f"\nOriginal Hard label: {labels} -> {classes[labels.item()]}\n")
    cropped_image, new_label = soft_crop(images[0], labels)
    soft_one_hot, loss = soft_target(
        pred=outputs, gold=new_label, reweight=reweight, soften_one_hot=soften_one_hot
    )

    resize_dim = 512
    cropped_pil_image = ff.to_pil_image(cropped_image).resize((resize_dim, resize_dim))
    cropped_image_path = (
        "/home/ekagra/Desktop/Study/MA/code/example/example_augmented_image.png"
    )
    cropped_pil_image.save(cropped_image_path)

    original_pil_image = ff.to_pil_image(images[0]).resize((resize_dim, resize_dim))
    original_image_path = (
        "/home/ekagra/Desktop/Study/MA/code/example/example_original_image.png"
    )
    original_pil_image.save(original_image_path)

    print(f"Softened Label Tensor: {soft_one_hot}\nUpdated New Loss: {loss}")
