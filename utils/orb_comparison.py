import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image


def orb_operation(im1, im2):
    if not isinstance(im1, Image.Image) or not isinstance(im2, Image.Image):
        pil = transforms.ToPILImage()
        im1 = pil(im1)
        im2 = pil(im2)
    im1 = im1.convert("L")
    im2 = im2.convert("L")
    im1_np = np.array(im1)
    im2_np = np.array(im2)

    orb = cv2.ORB_create(nfeatures=5000)
    keypoints1, descriptors1 = orb.detectAndCompute(im1_np, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_np, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    result = cv2.drawMatches(
        im1_np, keypoints1, im2_np, keypoints2, matches, im2_np, flags=2
    )

    # Display the best matching points
    plt.rcParams["figure.figsize"] = [14.0, 7.0]
    plt.title("Number of Matching Keypoints: " + str(len(matches)))
    plt.imshow(result)
    plt.show()

    # Print total number of matching points between the training and query images
    print(
        "\nNumber of Matching Keypoints Between The Training and Query Images: ",
        len(matches),
    )
    return len(matches)


def orb_correction_factor(
    original_image, augmented_image, display_matches: bool = False
):
    matches_reference = orb_operation(original_image, original_image)
    matches_12 = orb_operation(original_image, augmented_image)
    correction_factor = matches_12 / (matches_reference + 1e-5)
    return correction_factor


if __name__ == "__main__":

    im1_path = "/home/ekagra/Documents/GitHub/MasterArbeit/example/original_image.png"
    im2_path = "/home/ekagra/Documents/GitHub/MasterArbeit/example/augmented_image_geometric.png"
    # im2_path = '/home/ekagra/Documents/GitHub/MasterArbeit/example/original_image.png'
    im1 = Image.open(im1_path)
    im2 = Image.open(im2_path)

    resize = transforms.Resize(512)
    im1 = resize(im1)
    im2 = resize(im2)

    corr_fac = orb_correction_factor(original_image=im1, augmented_image=im2)
    print(f"Correction factor: {corr_fac:.3f}")
