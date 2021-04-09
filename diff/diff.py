import sys
import pandas as pd
import numpy as np
import cv2
import skimage as si

def main():
    # read args
    if len(sys.argv) < 3:
        print("Usage: python3 trabalho1.py <image1> <image2> [output_filename]")
        exit()

    name_img1 = str(sys.argv[1])
    name_img2 = str(sys.argv[2])
    if len(sys.argv) > 3:
        output_filename = str(sys.argv[3])
    else:
        output_filename = "./output.png"

    img1 = cv2.imread(name_img1)
    img2 = cv2.imread(name_img2)

    if len(img1.shape) > 2:
        img1 = img1[:,:,0]

    if len(img2.shape) > 2:
        img2 = img2[:,:,0]

    output = np.where(img1 - img2 < 0, (img1 - img2)*-1, img1 - img2)

    print(img1 - img2)

    cv2.imwrite(output_filename, output)

if __name__ == "__main__":
    main()
