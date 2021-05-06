import sys
import numpy as np
import cv2
import math
import argparse
from scipy.ndimage import rotate

#aplicando transformada de fourier na imagem
def fourier_transform(img):
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = 20*np.log(np.abs(fft_shift))

    return fft, fft_shift, magnitude_spectrum

#aplicando transformada inversa de fourier
def frequency_to_espacial_domain(fft, shifted=True):
    if shifted:
        ifftshift = np.fft.ifftshift(fft)
    else:
        ifftshift = fft
    espacial_img = np.fft.ifft2(ifftshift)

    return np.abs(espacial_img)

#criação dos kernels
def create_filters(img):
    rows, cols = img.shape[:2]
    image_center = (math.ceil(rows/2), math.ceil(cols/2))

    #kernel passa-baixa
    passa_baixa = np.zeros((rows, cols), np.uint8)
    cv2.circle(passa_baixa, image_center, 60, 1, cv2.FILLED)

    #kernel passa-faixa
    passa_faixa = np.zeros((rows, cols), np.uint8)
    cv2.circle(passa_faixa, image_center, 80, 1, cv2.FILLED)
    cv2.circle(passa_faixa, image_center, 20, 0, cv2.FILLED)

    #kernel passa-alta
    passa_alta = np.ones((rows, cols), np.uint8)
    cv2.circle(passa_alta, image_center, 75, 0, cv2.FILLED) 

    return passa_baixa, passa_faixa, passa_alta

#aplica filtro à imagem em domínio de frequência
def apply_filter(fft_shift, magnitude_spectrum, filter):
    filtered_spectrum = magnitude_spectrum * filter

    filtered = fft_shift * filter
    filtered_img = frequency_to_espacial_domain(filtered)

    return filtered_spectrum, filtered_img

#aplica compressão à imagem em domínio de frequência
def compress_img(fft, epsilon):
    magnitude_spectrum = np.log(np.abs(fft))
    fft[magnitude_spectrum < epsilon] = 0
    return frequency_to_espacial_domain(fft, shifted=False)

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

    fft, fft_shift, magnitude_spectrum = fourier_transform(img)
    cv2.imwrite(output_filename[:-4]+"_espectro_magnitude"+output_filename[-4:], magnitude_spectrum)
    inversa = frequency_to_espacial_domain(fft)
    cv2.imwrite(output_filename[:-4]+"_transformada_inversa"+output_filename[-4:], inversa)

    #aplicação dos filtros
    passa_baixa, passa_faixa, passa_alta = create_filters(img)

    filtered_spectrum, filtered_img = apply_filter(fft_shift, magnitude_spectrum, passa_baixa)
    cv2.imwrite(output_filename[:-4]+"_kernel_passa_baixa"+output_filename[-4:], filtered_spectrum)
    cv2.imwrite(output_filename[:-4]+"_passa_baixa"+output_filename[-4:], filtered_img)

    filtered_spectrum, filtered_img = apply_filter(fft_shift, magnitude_spectrum, passa_faixa)
    cv2.imwrite(output_filename[:-4]+"_kernel_passa_faixa"+output_filename[-4:], filtered_spectrum)
    cv2.imwrite(output_filename[:-4]+"_passa_faixa"+output_filename[-4:], filtered_img)

    filtered_spectrum, filtered_img = apply_filter(fft_shift, magnitude_spectrum, passa_alta)
    cv2.imwrite(output_filename[:-4]+"_kernel_passa_alta"+output_filename[-4:], filtered_spectrum)
    cv2.imwrite(output_filename[:-4]+"_passa_alta"+output_filename[-4:], filtered_img)

    #compressão
    compressed_img = compress_img(fft, compress_factor)
    cv2.imwrite(output_filename[:-4]+"_compressao"+output_filename[-4:], compressed_img)

    #rotação no domínio espacial
    rotated_img = rotated_img = rotate(img, rotate_angle, reshape=True)
    cv2.imwrite(output_filename[:-4]+"_rotacao"+output_filename[-4:], rotated_img)

    #espectro rotação
    fft, fft_shift, magnitude_spectrum = fourier_transform(rotated_img)
    cv2.imwrite(output_filename[:-4]+"_espectro_magnitude_rotacao"+output_filename[-4:], magnitude_spectrum)

if __name__ == "__main__":
    main()
