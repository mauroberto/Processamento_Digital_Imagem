import sys
import numpy as np
import cv2
import math
import argparse
from scipy.ndimage import rotate

#aplicando transformada de fourier na imagem
def fourier_transform(img):
    dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))

    return dft, dft_shift, magnitude_spectrum


#aplicando transformada reversa de fourier
def frequency_to_espacial_domain(dft, shifted=True):
    if shifted:
        ifftshift = np.fft.ifftshift(dft)
    else:
        ifftshift = dft
    espacial_img = cv2.idft(ifftshift)
    espacial_img = cv2.magnitude(espacial_img[:,:,0], espacial_img[:,:,1])
    espacial_img = cv2.normalize(espacial_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return espacial_img

#criação dos kernels
def create_filters(img):
    rows, cols = img.shape[:2]
    image_center = (rows//2, cols//2)

    #kernel passa-baixa
    passa_baixa = np.zeros((rows, cols), np.uint8)
    cv2.circle(passa_baixa, image_center, 80, 1, cv2.FILLED)

    #kernel passa-faixa
    passa_faixa = np.zeros((rows, cols), np.uint8)
    cv2.circle(passa_faixa, image_center, 100, 1, cv2.FILLED)
    cv2.circle(passa_faixa, image_center, 20, 0, cv2.FILLED)

    #kernel passa-alta
    passa_alta = np.ones((rows, cols), np.uint8)
    cv2.circle(passa_alta, image_center, 60, 0, cv2.FILLED) 

    return passa_baixa, passa_faixa, passa_alta

#aplica filtro à imagem em domínio de frequência
def apply_filter(dft_shift, magnitude_spectrum, filter):
    filtered_spectrum = cv2.bitwise_and(magnitude_spectrum, magnitude_spectrum, mask=filter)

    filtered = cv2.bitwise_and(dft_shift, dft_shift, mask=filter)
    filtered_img = frequency_to_espacial_domain(filtered)

    return filtered_spectrum, filtered_img

#rotaciona imagem no domínio de frequência
def rotate_img(img, angle):
    rotated_img = rotate(img, angle, reshape=True)
    return rotated_img

#aplica compressão à imagem em domínio de frequência
def compress_img(dft, epsilon):
    magnitude_spectrum = np.log(cv2.magnitude(dft[:,:,0], dft[:,:,1]))
    dft[abs(magnitude_spectrum) < epsilon] = 0
    return frequency_to_espacial_domain(dft, shifted=False)


def main():
    #read args
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', '-i', help="Image Path", type=str)
    parser.add_argument('--out', '-o', help="Output filename is optional. If not passed, the default value is './output.png'", type=str, default='./output.png')
    parser.add_argument('--angle', '-a', help="Rotate_angle] is optional. The default value is 45", type=float, default=45.0)
    parser.add_argument('--compress', '-c', help="Compress factor is optional. The default value is 105", type=float, default=10.5)

    if len(sys.argv) < 2:
        print(parser.format_help())
        exit()

    args = parser.parse_args(sys.argv[1:])

    try:
        img_name = args.img
        output_filename = args.out
        rotate_angle = args.angle
        compress_factor = args.compress
    except:
        print(parser.format_help())
        exit()

    #read image
    img = cv2.imread(img_name, 0)

    dft, dft_shift, magnitude_spectrum = fourier_transform(img)
    cv2.imwrite(output_filename[:-4]+"_espectro_magnitude"+output_filename[-4:], magnitude_spectrum)


    #aplicação dos filtros
    passa_baixa, passa_faixa, passa_alta = create_filters(img)

    filtered_spectrum, filtered_img = apply_filter(dft_shift, magnitude_spectrum, passa_baixa)
    cv2.imwrite(output_filename[:-4]+"_kernel_passa_baixa"+output_filename[-4:], filtered_spectrum)
    cv2.imwrite(output_filename[:-4]+"_passa_baixa"+output_filename[-4:], filtered_img)

    filtered_spectrum, filtered_img = apply_filter(dft_shift, magnitude_spectrum, passa_faixa)
    cv2.imwrite(output_filename[:-4]+"_kernel_passa_faixa"+output_filename[-4:], filtered_spectrum)
    cv2.imwrite(output_filename[:-4]+"_passa_faixa"+output_filename[-4:], filtered_img)

    filtered_spectrum, filtered_img = apply_filter(dft_shift, magnitude_spectrum, passa_alta)
    cv2.imwrite(output_filename[:-4]+"_kernel_passa_alta"+output_filename[-4:], filtered_spectrum)
    cv2.imwrite(output_filename[:-4]+"_passa_alta"+output_filename[-4:], filtered_img)

    #compressão
    compressed_img = compress_img(dft, compress_factor)
    cv2.imwrite(output_filename[:-4]+"_compressao"+output_filename[-4:], compressed_img)

    #rotação
    rotated_img = rotate_img(img, rotate_angle)
    cv2.imwrite(output_filename[:-4]+"_rotacao"+output_filename[-4:], rotated_img)

    #espectro rotação
    dft, dft_shift, magnitude_spectrum = fourier_transform(rotated_img)
    cv2.imwrite(output_filename[:-4]+"_espectro_magnitude_rotacao"+output_filename[-4:], magnitude_spectrum)

if __name__ == "__main__":
    main()
