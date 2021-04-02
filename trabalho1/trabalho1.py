import sys
import pandas as pd
import numpy as np
import cv2
import skimage

#item A
def filterA(img):
    b,g,r = cv2.split(img)
    filter = np.array([
                       [0.393, 0.769, 0.189],
                       [0.349, 0.686, 0.168],
                       [0.272, 0.534, 0.131]
                    ])

    _r = (img.T * filter[0][:, None, None]).sum(axis=0).T
    _g = (img.T * filter[1][:, None, None]).sum(axis=0).T
    _b = (img.T * filter[2][:, None, None]).sum(axis=0).T

    _r[_r > 255] = 255
    _g[_g > 255] = 255
    _b[_b > 255] = 255

    return cv2.merge((_b, _g, _r)).astype(np.uint8)


#item B
def filterB(img):
    filter = np.array([0.2989, 0.5870, 0.1140])

    I = (img.T * filter[:, None, None]).sum(axis=0).T

    I[I > 255] = 255

    return I

#defining filters
h1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
h2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
h3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
h4 = np.ones((3, 3))/9
h5 = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]])
h6 = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
h7 = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
h8 = np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]])
h9 = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])/256

filters = np.array([h1, h2, h3, h4, h5, h6, h7, h8, h9], dtype=object)

#apply convolution to image
def applyFilter(image, filter):
    #convert 3d grayscale images to 2d 
    if len(image.shape) > 2:
        image = image[:,:,0]
    rows, cols = image.shape[:2]
    rows_k, cols_k = filter.shape[:2]

    pad_c = cols_k//2
    pad_r = rows_k//2

    # creating output array
    output_array = np.zeros((rows, cols))
    
    # adding padding to image
    padded_image = np.pad(image, [
        (pad_r, pad_r),
        (pad_c, pad_c)
    ])

    output_array_rows, output_array_cols = output_array.shape[:2]

    # non-vectorized version
    #for r in range(output_array_rows):
    #    for c in range(output_array_cols):
    #        output_array[r, c] = (filter * padded_image[r:r + rows_k, c:c + cols_k]).sum()

    # vectorized version
    # dividing padded_image into sub-matrices of the filter size
    sub_matrices = skimage.util.shape.view_as_windows(padded_image, (rows_k, cols_k))
    output_array = np.einsum('ij,rcij->rc', filter, sub_matrices)
    
    return output_array

#combine two filters
def combineFilters(img, filter1, filter2):
    img1 = applyFilter(img, filter1).astype(np.float32)
    img2 = applyFilter(img, filter2).astype(np.float32)
    img1 = np.power(img1,2)
    img2 = np.power(img2,2)
    img3 = np.around(np.sqrt(img1 + img2))
    img3 = cv2.normalize(img3, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
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

    #select filter
    if type_of_img == 0:
        if flt_type < 0 or flt_type > 2:
            print("Invalid filter. The filter value must be an integer on interval [0, 2]")
            exit()
        elif flt_type == 1:
            output_img = filterA(img)
        elif flt_type == 2:
            output_img = filterB(img)
        else:
            output_img = filterA(img)
            cv2.imwrite(output_filename[:-4]+"_A"+output_filename[-4:], output_img)
            output_img = filterB(img)
            cv2.imwrite(output_filename[:-4]+"_B"+output_filename[-4:], output_img)
            exit()
    else:
        if flt_type < 0 or flt_type > 10:
            print("Invalid filter. The filter value must be an integer on interval [0, 10]")
            exit()
        elif flt_type == 0:
            for i in range(0, 9):
                output_img = applyFilter(img, filters[i])
                cv2.imwrite(output_filename[:-4]+"_h"+str(i+1)+output_filename[-4:], output_img)
            output_img = combineFilters(img, filters[0], filters[1])
            cv2.imwrite(output_filename[:-4]+"_h1_and_h2"+output_filename[-4:], output_img)
            exit()
        elif flt_type == 10:
            output_img = combineFilters(img, filters[0], filters[1])
        else:
            output_img = applyFilter(img, filters[flt_type-1])

    #write output image
    cv2.imwrite(output_filename, output_img)

if __name__ == "__main__":
    main()
