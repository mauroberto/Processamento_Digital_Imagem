import sys
import numpy as np
import cv2
import skimage as si



def main():
    # read args
    if len(sys.argv) < 2:
        print("Usage: python3 trabalho2.py <image_path> [output_filename]")
        print("[output_filename] is optional. If not passed, the default value is './output.png'")
        exit()

    img_name = str(sys.argv[1])

    if len(sys.argv) > 4:
        output_filename = str(sys.argv[4])
    else:
        output_filename = "./output.png"

    # read image
    img = cv2.imread(img_name, 0)


    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    cv2.imwrite(output_filename, magnitude_spectrum)

if __name__ == "__main__":
    main()
