import sys
import pandas as pd
import numpy as np
import cv2

#item A
def filter_A(img):
    b,g,r = cv2.split(img)

    print(r.shape)
    print(g.shape)
    print(b.shape)

    _r = np.where(r*0.393 + g*0.769 + b*0.189 > 255, 255, r*0.393 + g*0.769 + b*0.189)
    _g = np.where(r*0.349 + g*0.686 + b*0.168 > 255, 255, r*0.349 + g*0.686 + b*0.168)
    _b = np.where(r*0.272 + g*0.534 + b*0.131 > 255, 255, r*0.272 + g*0.534 + b*0.131)

    print(_r.shape)
    print(_g.shape)
    print(_b.shape)

    img2 = cv2.merge((_b, _g, _r)).astype(np.uint8)

    print(img2)

    cv2.imshow('image_A', img2)
    cv2.waitKey(0)

    return img2


#item B
def filter_B(img):
    b,g,r = cv2.split(img)

    img2 = np.where(r*0.2989 + g*0.5870 + b*0.1140 > 255, 255, r*0.2989 + g*0.5870 + b*0.1140)

    print(img2)

    cv2.imshow('image_B', img2)
    cv2.waitKey(0)

    return img2

#filters
h1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
h2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
h3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
h4 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])/9
h5 = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]])
h6 = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
h7 = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
h8 = np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]])
h9 = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])/256

filters = np.array([h1, h2, h3, h4, h5, h6, h7, h8, h9])

def apply_filter(img, filter):
    img2 = cv2.filter2D(img,-1,filter)
    cv2.imshow('filter h', img2)
    cv2.waitKey(0)
    return img2

def combineFilters(img, filter1, filter2):
    img1 = apply_filter(img, filter1).astype(np.float32)
    img2 = apply_filter(img, filter2).astype(np.float32)
    img1 = np.power(img1,2)
    img2 = np.power(img2,2)
    img3 = np.around(np.sqrt(img1 + img2))
    img3 = cv2.normalize(img3, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imshow('combine filters h', img3)
    cv2.waitKey(0)
    return img3

def main():
    #read args
    if len(sys.argv) < 4:
        print("Usage: python3 trabalho1.py <image_path> <type_of_img> <filter_type> [output_filename]")
        print("<type_of_img> must be 0 if was passed a colored image and 1 otherwise")
        print("<filter_type>:")
        print("               In case of monochromatic image, pass 1 to execute filter of item A, 2 to filter of item B or 0 to execute A and B")
        print("               In case of colored image, pass 'i' to execute filter h_i, 0 to execute all the filters or 10 to combine filters h_1 and h_2")
        print("[output_filename] is optional. If not passed, the default value is './output.png'")
        exit()

    img_name = str(sys.argv[1])
    type_of_img = int(sys.argv[2])
    flt_type = int(sys.argv[3])

    if len(sys.argv) > 4:
        output_filename = str(sys.argv[4])
    else:
        output_filename = "./output.png"

    #read image
    img = cv2.imread(img_name)

    cv2.imshow('image', img)
    cv2.waitKey(0)

    if type_of_img == 0:
        if flt_type < 0 or flt_type > 2:
            print("Invalid filter. The filter value must be an integer on interval [0, 2]")
            exit()
        elif flt_type == 1:
            output_img = filter_A(img)
        elif flt_type == 2:
            output_img = filter_B(img)
        else:
            output_img = filter_A(img)
            cv2.imwrite(output_filename[:-4]+"_A"+output_filename[-4:], output_img)
            output_img = filter_B(img)
            cv2.imwrite(output_filename[:-4]+"_B"+output_filename[-4:], output_img)
            exit()
    else:
        if flt_type < 0 or flt_type > 10:
            print("Invalid filter. The filter value must be an integer on interval [0, 10]")
            exit()
        elif flt_type == 0:
            for i in range(0, 9):
                output_img = apply_filter(img, filters[i])
                cv2.imwrite(output_filename[:-4]+"_h"+str(i+1)+output_filename[-4:], output_img)
            output_img = combineFilters(img, filters[0], filters[1])
            cv2.imwrite(output_filename[:-4]+"_h1_and_h2"+output_filename[-4:], output_img)
            exit()
        elif flt_type == 10:
            output_img = combineFilters(img, filters[0], filters[1])
        else:
            output_img = apply_filter(img, filters[i-1])

    cv2.imwrite(output_filename, output_img)

if __name__ == "__main__":
    main()
