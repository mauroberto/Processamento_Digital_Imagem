import os

imgs = ["milho1.png", "milho2.png", "textura1.png", "textura2.png", "textura3.png", "textura4.png"]

for i in range(0, len(imgs)-1):
    img1 = imgs[i]
    for j in range(i+1, len(imgs)):
        img2 = imgs[j]
        os.system("python3 trabalho4.py -i imagens/%s -j imagens/%s -o output_images/%s_%s.png" % (img1, img2, img1[:-4], img2[:-4]))

