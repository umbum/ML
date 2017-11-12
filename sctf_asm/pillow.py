#-*-coding: utf-8 -*-
from PIL import Image, ImageFilter

with Image.open('D:\\Github\\ml\\ImageFiltering\\st1.png') as img:
    #blurImage = im.filter(ImageFilter.Kernel((3, 3), )
    #blurImage.save("D:\\Github\\ml\\ImageFiltering\\blured.png")
    
    '''
    #직접 픽셀에 접근하려면
    pix = img.load()
    for x in range(10):
        pix[x, x] = (0, 0, 0)
    '''

    # 이미지 반전
    img = Image.eval(im, lambda x : 256 - x)
    img.save("D:\\Github\\ml\\ImageFiltering\\im.png")