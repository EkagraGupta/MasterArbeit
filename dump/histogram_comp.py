from PIL import ImageChops, Image
import matplotlib.pyplot as plt
import numpy as np

actual_error = 0
im1 = Image.open("/home/ekagra/Desktop/Study/MA/code/example/original_image.png")
x = np.array(im1.histogram())

im2 = Image.open("/home/ekagra/Desktop/Study/MA/code/example/augmented_image.png")
y = np.array(im2.histogram())

try:
    if len(x) == len(y):
        error = np.sqrt(((x - y) ** 2).mean())
        error = str(error)[:2]
        actual_error = float(100) - float(error)
    diff = ImageChops.difference(im1, im2).getbbox()
    if diff:
        print("Not Duplicate Image")
        print("Matching Images In percentage: ", actual_error, "\t%")
        f = plt.figure()
        text_lable = str("Matching Images Percentage " + str(actual_error) + "%")
        plt.suptitle(text_lable)
        f.add_subplot(1, 2, 1)
        plt.imshow(im1)
        f.add_subplot(1, 2, 2)
        plt.imshow(im2)
        plt.show(block=True)
    else:
        print("Duplicate Image")
        print("Matching Images In percentage: ", actual_error, "%")
        f = plt.figure()
        text_lable = str("Matching Images Percentage" + str(actual_error) + "%")
        plt.suptitle(text_lable)
        f.add_subplot(1, 2, 1)
        plt.imshow(im1)
        f.add_subplot(1, 2, 2)
        plt.imshow(im2)
        plt.show(block=True)

except ValueError as identifier:
    f = plt.figure()
    text_lable = str("Matching Images Percentage " + str(actual_error) + "%")
    plt.suptitle(text_lable)
    f.add_subplot(1, 2, 1)
    plt.imshow(im1)
    f.add_subplot(1, 2, 2)
    plt.imshow(im2)
    plt.show(block=True)
    print("identifier: ", identifier)
