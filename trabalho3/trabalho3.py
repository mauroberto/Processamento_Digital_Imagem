import sys
import numpy as np
import cv2
import argparse
import random as rand
import matplotlib.pyplot as plt

rand.seed(192800)

def histogram(areas, output_filename):
    n, bins, patches = plt.hist(areas, bins=[min(areas), 1500, 3000, max(areas)], edgecolor='black', linewidth=1)
    plt.xlabel("Área")
    plt.ylabel("Número de Objetos")
    plt.savefig(output_filename[:-4]+"_histograma"+output_filename[-4:])

    print("número de regiões pequenas: %d" % n[0])
    print("número de regiões médias: %d" % n[1])
    print("número de regiões grandes: %d" % n[2])


def getCentroid(cnt):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    return (cx, cy)

def getSolidity(cnt, area):
    #convex hull
    convex_hull = cv2.convexHull(cnt)

    #convex hull area
    convex_area = cv2.contourArea(convex_hull)

    #solidity = contour area / convex hull area
    return area/float(convex_area)

def getEccentricity(cnt):
    ellipse = cv2.fitEllipse(cnt)

    #center, axis_length and orientation of ellipse
    (center, axes, orientation) = ellipse

    #length of major and minor axis
    majoraxis_length = max(axes)
    minoraxis_length = min(axes)

    #eccentricity
    return np.sqrt(1-(minoraxis_length/majoraxis_length)**2)

def drawContours(contours, edged, hierarchy):
    drawing = np.full((edged.shape[0], edged.shape[1], 3), 255, dtype=np.uint8)
    for i in range(len(contours)):
        color = (0, 0, 255)
        cv2.drawContours(drawing, contours, i, color, 1, cv2.LINE_8, hierarchy, 0)

    return drawing

def drawContoursFilled(contours, edged):
    drawing = np.full((edged.shape[0], edged.shape[1], 3), 255, dtype=np.uint8)
    for i in range(len(contours)):
        cnt = contours[i]
        color = (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255))
        cv2.fillPoly(drawing, pts = [cnt], color=color)
        cv2.putText(drawing, str(i), getCentroid(cnt), cv2.FONT_HERSHEY_PLAIN, 1, 0)

    return drawing

def extractProperties(contours):
    areas = np.zeros((len(contours)))

    print("número de regiões: %d" % len(contours))

    for i in range(0, len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        areas[i] = area
        print("região %d:" % (i), end=" ")
        print("área: %f" % area, end=" ")
        print("perímetro: %f" % cv2.arcLength(cnt,True), end=" ")
        print("excentricidade: %f" % getEccentricity(cnt), end=" ")
        print("solidez: %f" % getSolidity(cnt, area))

    return areas

def main():
    #read args
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

    #read image as grayscale
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

    #converting image to black and white
    (thresh, im_bw) = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)
    cv2.imwrite(output_filename[:-4]+"_binaria"+output_filename[-4:], im_bw)

    edged = cv2.Canny(im_bw, 30, 200)
    _, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imwrite(output_filename[:-4]+"_thresh"+output_filename[-4:], edged)
    
    #1.2 - draw contours
    drawing = drawContours(contours, edged, hierarchy)
    cv2.imwrite(output_filename[:-4]+"_contornos"+output_filename[-4:], drawing)

    #1.3 - extracting and printing properties from contours
    areas = extractProperties(contours)

    #1.3 - draw contours, fill and label them
    drawing2 = drawContoursFilled(contours, edged)
    cv2.imwrite(output_filename[:-4]+"_contornos_preenchidos"+output_filename[-4:], drawing2)

    #1.4 - histogram
    histogram(areas, output_filename)


if __name__ == "__main__":
    main()
