import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms


def sift_operation(im1, im2, display_matches: bool = False):
    if not isinstance(im1, Image.Image) or not isinstance(im2, Image.Image):
        pil = transforms.ToPILImage()
        im1 = pil(im1)
        im2 = pil(im2)
    im1 = im1.convert("L")
    im2 = im2.convert("L")
    im1_np = np.array(im1)
    im2_np = np.array(im2)
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(im1_np, None)
    keypoints2, descriptors2 = sift.detectAndCompute(im2_np, None)

    if descriptors1 is None or descriptors2 is None:
        # print(
        #     "Either the images are too different or lacking sufficient features for SIFT to detect"
        # )
        # resize = transforms.Resize(512)
        # if descriptors1 is None:
        #     im1_resize = resize(im1)
        #     im1_resize.show()
        # if descriptors2 is None:
        #     im2_resize = resize(im2)
        #     im2_resize.show()
        return 1

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    # print(f"Num matches: {len(matches)}")

    if display_matches:
        im3 = cv2.drawMatches(
            im1_np, keypoints1, im2_np, keypoints2, matches[:500], im2_np, flags=2
        )
        plt.imshow(im3)
        plt.imsave(
            "/home/ekagra/Desktop/Study/MA/code/example/sift_test_example4.png", im3
        )
        plt.show()
    print(f'Number of Matching Keypoints Between the Traning and Query Images: {len(matches)}')
    return len(matches)


def sift_correction_factor(
    original_image, augmented_image, display_matches: bool = False
):
    matches_reference = sift_operation(original_image, original_image)
    matches_12 = sift_operation(original_image, augmented_image, display_matches)
    correction_factor = matches_12 / matches_reference
    return correction_factor


if __name__ == "__main__":
    im1_path = "/home/ekagra/Documents/GitHub/MasterArbeit/example/original_image.png"
    im2_path = "/home/ekagra/Documents/GitHub/MasterArbeit/example/augmented_image.png"
    im1 = Image.open(im1_path)
    im2 = Image.open(im2_path)
    im1_gray = im1.convert("L")
    im2_gray = im2.convert("L")

    # matches = sift_operation(im1=im1_gray, im2=im2_gray, display_matches=True)
    corr_fac = sift_correction_factor(
        original_image=im1, augmented_image=im2, display_matches=True
    )
    print(f"Correction factor: {corr_fac}")
