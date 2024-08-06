from pixelmatch.contrib.PIL import pixelmatch
from PIL import Image

im1_path = "/home/ekagra/Documents/GitHub/MasterArbeit/example/original_image.png"
im2_path = "/home/ekagra/Documents/GitHub/MasterArbeit/example/augmented_image.png"

im1 = Image.open(im1_path)
im2 = Image.open(im2_path)
im_diff = Image.new("RGBA", im1.size)

px_diff = pixelmatch(im1, im2, im_diff, includeAA=True)

print(px_diff)
