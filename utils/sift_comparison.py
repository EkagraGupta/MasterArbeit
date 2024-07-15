import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im1_path = "/home/ekagra/Desktop/Study/MA/code/example/resized_example_image.png"
im2_path = "/home/ekagra/Desktop/Study/MA/code/example/augmented_example_image.png"

im1 = Image.open(im1_path)
im2 = Image.open(im2_path)
im1_np = np.array(im1)
im2_np = np.array(im2)
gray1 = np.array(im1.convert("L"))
gray2 = np.array(im2.convert("L"))

sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(im1_np, None)
keypoints2, descriptors2 = sift.detectAndCompute(im2_np, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

print(f"Num Matches: {len(matches)}")

im3 = cv2.drawMatches(
    im1_np, keypoints1, im2_np, keypoints2, matches[:500], im2_np, flags=2
)
plt.imshow(im3)
plt.show()
