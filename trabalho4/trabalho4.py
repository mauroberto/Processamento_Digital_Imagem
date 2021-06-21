import sys
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, texture
from skimage.measure import shannon_entropy
import skimage


def getNameFromPath(path):
    if path[-4:-3] == ".":
        path = path[:-4]

    if "/" in path:
        return path.split("/")[-1]
    return path

def plotImages(img1, img2, name, title1, title2):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    plt.gray()
    fig.tight_layout()
    ax1.imshow(img1)
    ax2.imshow(img2)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax1.title.set_text(title1)
    ax2.title.set_text(title2)
    plt.savefig(name)
    plt.cla()
    plt.clf()
    plt.close() 

def glcm(img, label="imagem 1"):
    distances = [5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    label_angles = ["0", "45", "90", "135"]
    
    _glcm = texture.greycomatrix(img, distances, angles, 256)
    contrast = texture.greycoprops(_glcm, 'contrast')
    ASM = texture.greycoprops(_glcm, 'ASM')

    for distance_idx in range(0, len(distances)):
        print("Distância: %f" % distances[distance_idx])

        for angle_idx in range(0, len(angles)):
            print("Ângulo: %s" % label_angles[angle_idx])
            print("Contraste %s: %.4f" % (label, contrast[distance_idx, angle_idx]))
            print("Segundo momento angular %s: %f" % (label, ASM[distance_idx, angle_idx]))
            print("Entropia %s: %.4f" % (label, shannon_entropy(_glcm[:, :, distance_idx, angle_idx])))

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
        img_path = args.img
        img_name = getNameFromPath(img_path)
        img2_path = args.img2
        img2_name = getNameFromPath(img2_path)
        output_filename = args.out
    except: 
        print(parser.format_help())
        exit()

    # lendo imagens em tons de cinza
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    plotImages(img, img2, output_filename[:-4]+"_grayscale"+output_filename[-4:], img_name, img2_name)

    # LBP
    radius = 1
    n_points = 8
    lbp_img = local_binary_pattern(img, n_points, radius)
    lbp_img2 = local_binary_pattern(img2, n_points, radius)
    plotImages(lbp_img, lbp_img2, output_filename[:-4]+"_LBP"+output_filename[-4:], img_name, img2_name)

    # criando histogramas
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 5))
    n, bins, patches = ax1.hist(lbp_img.ravel(), density=True, bins=256, range=(0, 256))
    n2, bins2, patches2 = ax2.hist(lbp_img2.ravel(), density=True, bins=256, range=(0, 256))
    ax1.title.set_text(img_name)
    ax2.title.set_text(img2_name)
    plt.savefig(output_filename[:-4]+"_histograma"+output_filename[-4:])
    plt.cla()
    plt.clf()
    plt.close()

    # métodos de comparação do OpenCV
    CV2_METHODS = (
        ("Correlação", cv2.HISTCMP_CORREL),
        ("Qui-Quadrado", cv2.HISTCMP_CHISQR),
        ("Interseção", cv2.HISTCMP_INTERSECT),
        ("Bhattacharyya", cv2.HISTCMP_BHATTACHARYYA)
    )

    for (name, compare_method) in CV2_METHODS:
        value = cv2.compareHist(np.float32(n), np.float32(n2), compare_method)
        print("%s: %.4f" % (name, value))
    

    # GLCM
    glcm(img)
    glcm(img2, "imagem 2")

if __name__ == "__main__":
    main()
