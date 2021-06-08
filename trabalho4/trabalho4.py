import sys
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from skimage.measure import shannon_entropy

def main():
    #lendo argumentos
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', '-i', help="Image Path", type=str)
    parser.add_argument('--img2', '-j', help="Image2 Path", type=str)
    parser.add_argument('--out', '-o', help="Output filename is optional. If not passed, the default value is './output.png'", type=str, default='./output.png')

    if len(sys.argv) < 2:
        print(parser.format_help())
        exit()

    args = parser.parse_args(sys.argv[1:])

    try:
        img_name = args.img
        img2_name = args.img2
        output_filename = args.out
    except: 
        print(parser.format_help())
        exit()

    #lendo imagens em tons de cinza
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(output_filename[:-4]+"_grayscale"+output_filename[-4:], img)

    img2 = cv2.imread(img2_name, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(output_filename[:-4]+"_grayscale"+output_filename[-4:], img2)


    radius = 6
    n_points = 8 * radius
    lbp_img = local_binary_pattern(img, n_points, radius, 'uniform')
    cv2.imwrite(output_filename[:-4]+"_LBP"+output_filename[-4:], lbp_img)

    lbp_img2 = local_binary_pattern(img2, n_points, radius, 'uniform')
    cv2.imwrite(output_filename[:-4]+"_LBP_img2"+output_filename[-4:], lbp_img2)


    n_bins = int(lbp_img.max() - lbp_img.min() + 1)
    n, bins, patches = plt.hist(lbp_img.ravel(), density=True, bins=n_bins, range=(lbp_img.min(), lbp_img.max()+1))
    plt.yscale('log', nonpositive='clip')
    plt.savefig(output_filename[:-4]+"_histograma_img2"+output_filename[-4:])
    plt.cla()

    n_bins2 = int(lbp_img2.max() - lbp_img2.min() + 1)
    n2, bins2, patches2 = plt.hist(lbp_img2.ravel(), density=True, bins=n_bins2, range=(lbp_img2.min(), lbp_img2.max() + 1))
    plt.yscale('log', nonpositive='clip')
    plt.savefig(output_filename[:-4]+"_histograma"+output_filename[-4:])

    # initialize OpenCV methods for histogram comparison
    OPENCV_METHODS = (
        ("Correlação", cv2.HISTCMP_CORREL),
        ("Chi-Quadrado", cv2.HISTCMP_CHISQR),
        ("Intersecção", cv2.HISTCMP_INTERSECT),
        ("Bhattacharyya", cv2.HISTCMP_BHATTACHARYYA)
    )


    for (name, compare_method) in OPENCV_METHODS:
        value = cv2.compareHist(np.float32(n), np.float32(n2), compare_method)
        print(name, ":", value)
    


    #GLCM

    glcm = greycomatrix(img, [5], [0], 256, True, True)
    print("Contraste imagem 1:", greycoprops(glcm, 'contrast')[0, 0])
    print("Segundo momento angular imagem 1:", greycoprops(glcm, 'ASM')[0, 0])
    print("Entropia imagem 1:", shannon_entropy(glcm[:, :, 0, 0]))

    glcm2 = greycomatrix(img2, [5], [0], 256, True, True)
    print("Contraste imagem 2:", greycoprops(glcm2, 'contrast')[0, 0])
    print("Segundo momento angular imagem 2:", greycoprops(glcm2, 'ASM')[0, 0])
    print("Entropia imagem 2:", shannon_entropy(glcm2[:, :, 0, 0]))


if __name__ == "__main__":
    main()
