import sys
import numpy as np
import cv2
import argparse
import random as rand
import matplotlib.pyplot as plt

#seed para a gerar cores aleatórias
rand.seed(192800)

def histogram(areas, output_filename):
    n, bins, patches = plt.hist(areas, bins=[min(min(areas), 0), 1500, 3000, max(max(areas), 4000)], edgecolor='black', linewidth=1)
    plt.xlabel("Área")
    plt.ylabel("Número de Objetos")
    plt.savefig(output_filename[:-4]+"_histograma"+output_filename[-4:])

    print("número de regiões pequenas: %d" % n[0])
    print("número de regiões médias: %d" % n[1])
    print("número de regiões grandes: %d" % n[2])


def getCentroid(cnt):
    M = cv2.moments(cnt)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    else:
        cx, cy = cnt[0][0]

    return (cx, cy)

def getSolidity(cnt, area):
    #fecho convexo
    convex_hull = cv2.convexHull(cnt)

    #area do fecho convexo
    convex_area = cv2.contourArea(convex_hull)

    if convex_area == 0:
        return 0.0

    #solidez = area / area do fecho convexo
    return area/float(convex_area)

def getEccentricity(cnt):
    if cnt.shape[0] < 5:
        return 0.0 

    ellipse = cv2.fitEllipse(cnt)

    #centro, tamanho dos eixos e orientação da elipse
    (center, axis, orientation) = ellipse

    #comprimento do maior e menor eixo
    max_axle = max(axis)
    min_axle = min(axis)

    return np.sqrt(1-(min_axle/max_axle)**2)

def drawContours(contours, edged, hierarchy):
    drawing = np.full((edged.shape[0], edged.shape[1], 3), 255, dtype=np.uint8)
    color = (0, 0, 255)
    cv2.drawContours(drawing, contours, -1, color, 1)

    return drawing

def drawContoursFilled(contours, edged):
    drawing = np.full((edged.shape[0], edged.shape[1], 3), 255, dtype=np.uint8)
    for i in range(0, len(contours)):
        cnt = contours[i]
        color = (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255))
        cv2.fillPoly(drawing, pts = [cnt], color=color)
        (cX, cY) = getCentroid(cnt)
        cv2.putText(drawing, str(i), (cX - 6, cY + 6), cv2.FONT_HERSHEY_PLAIN, 1, 2)

    return drawing

def extractProperties(contours):
    areas = np.zeros((len(contours)))

    print("número de regiões: %d" % len(contours), end="\n\n")

    for i in range(0, len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        areas[i] = area
        print("região %d:" % (i), end=" ")
        print("área: %.0f" % area, end=" ")
        print("perímetro: %f" % cv2.arcLength(cnt,True), end=" ")
        print("excentricidade: %f" % getEccentricity(cnt), end=" ")
        print("solidez: %f" % getSolidity(cnt, area))

    print("")

    return areas

def filterContoursByHierarchy(contours, hierarchy):
    cntours = []

    for i in range(0, len(contours)):
        if hierarchy[0][i][3] == -1:
            continue
        cntours.append(contours[i])

    return cntours

def main():
    #lendo argumentos
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', '-i', help="Image Path", type=str)
    parser.add_argument('--out', '-o', help="Output filename is optional. If not passed, the default value is './output.png'", type=str, default='./output.png')

    if len(sys.argv) < 2:
        print(parser.format_help())
        exit()

    args = parser.parse_args(sys.argv[1:])

    try:
        img_name = args.img
        output_filename = args.out
    except:
        print(parser.format_help())
        exit()

    #lendo imagem em tons de cinza
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

    #convertendo imagem para preto e branco
    im_bw = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite(output_filename[:-4]+"_binaria"+output_filename[-4:], im_bw)

    laplacian_im_bw = cv2.Laplacian(im_bw, cv2.CV_8U, ksize=3)

    _, contours, hierarchy = cv2.findContours(laplacian_im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #1.2 - desenhando contornos
    drawing = drawContours(contours, im_bw, hierarchy)
    cv2.imwrite(output_filename[:-4]+"_contornos"+output_filename[-4:], drawing)

    #1.3 - extraindo e imprimindo as propriedades dos contornos
    areas = extractProperties(contours)

    #1.3 - desenhando contornos preenchidos e com rótulos
    drawing2 = drawContoursFilled(contours, im_bw)
    cv2.imwrite(output_filename[:-4]+"_contornos_preenchidos"+output_filename[-4:], drawing2)

    #1.4 - histograma de área dos objetos
    histogram(areas, output_filename)


if __name__ == "__main__":
    main()
