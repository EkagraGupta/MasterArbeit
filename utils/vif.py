import piq
import numpy as np
from torchvision import transforms
from PIL import Image

def compute_vif(im1_tensor, im2_tensor):
    im1_tensor /= 255.
    im2_tensor /= 255.
    
    vif_value = piq.vif_p(im1_tensor, im2_tensor)
    return np.clip(vif_value, 0., 1.)    


if __name__=='__main__':
    im1_path = '/home/ekagra/Documents/GitHub/MasterArbeit/example/original_example_image.png'
    im2_path = '/home/ekagra/Documents/GitHub/MasterArbeit/example/augmented_image.png'

    to_tensor = transforms.ToTensor()

    im1_tensor = to_tensor(Image.open(im1_path)).unsqueeze(0)
    im2_tensor = to_tensor(Image.open(im2_path)).unsqueeze(0)
    print(im1_tensor.shape)

    vif_val = compute_vif(im1_tensor, im2_tensor)
    print(vif_val)

