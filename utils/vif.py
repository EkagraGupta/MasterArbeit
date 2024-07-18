import piq
from torchvision import transforms


def vif(original_image, augmented_image):
    resize = transforms.Resize((41, 41))
    # to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    # im_resize = to_tensor(resize(to_pil(original_image)))
    # augment_im_resize = to_tensor(resize(to_pil(augmented_image)))
    im_resize = to_tensor(resize(original_image))
    augment_im_resize = to_tensor(resize(augmented_image))
    vif_value = piq.vif_p(im_resize.unsqueeze(0), augment_im_resize.unsqueeze(0))
    # print(f"VIF: {vif_value}")
    return vif_value
