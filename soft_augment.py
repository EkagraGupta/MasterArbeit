import torch

from torchvision import transforms
from torchvision.transforms import functional as ff

from dataset import load_dataset


class SoftAugment:
    def __init__(
        self,
        n_class=10,
        sigma_crop=10,
        t_crop=1.0,
        max_p_crop=1.0,
        pow_crop=4.0,
        bg_crop=0.01,
        k=2,
    ):
        self.n_class = n_class
        self.chance = 1 / n_class
        self.k = k

        # crop parameters
        self.sigma_crop = sigma_crop
        self.t_crop = t_crop
        self.max_p_crop = max_p_crop
        self.pow_crop = pow_crop
        self.bg_crop = bg_crop

    def draw_offset(self, sigma=0.3, limit=24, n=100):
        # draw an integer from a (clipped) Gaussian
        for d in range(n):
            x = torch.randn((1)) * sigma
            if abs(x) <= limit:
                return int(x)
        return int(0)

    def compute_visibility(self, dim1, dim2, tx, ty):
        return (dim1 - abs(tx)) * (dim2 - abs(ty)) / (dim1 * dim2)

    def __call__(self, image):
        dim1, dim2 = image.size(1), image.size(2)

        # create background
        bg = torch.ones((3, dim1 * 3, dim2 * 3)) * self.bg_crop * torch.randn((3, 1, 1))
        bg[:, dim1 : dim1 * 2, dim2 : dim2 * 2] = image  # put image at the center

        tx, ty = self.draw_offset(self.sigma_crop, dim1), self.draw_offset(
            self.sigma_crop, dim2
        )

        left, right = tx + dim1, tx + dim1 * 2
        top, bottom = ty + dim2, ty + dim2 * 2
        # print(f'l: {left}\tr: {right}\tt: {top}\tb: {bottom}\nd1: {bg.size(1)}\td2: {bg.size(2)}')

        visibility = self.compute_visibility(dim1, dim2, tx, ty)
        confidence = 1 - (1 - self.chance) * (1 - visibility) ** self.k

        cropped_image = bg[:, left:right, top:bottom]
        return cropped_image, confidence


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    trainloader, _, classes = load_dataset(batch_size=1, transform=transform)
    images, labels = next(iter(trainloader))
    print(f"\nOriginal Hard label: {labels} -> {classes[labels.item()]}\n")
    soft_augment = SoftAugment()

    cropped_image, confidence = soft_augment(images[0])
    pil_cropped_image = ff.to_pil_image(cropped_image)
    pil_cropped_image.save(
        "/home/ekagra/Desktop/Study/MA/code/example/example_augmented_image.png"
    )
