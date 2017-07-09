import cv2
import numpy as np

def _main():
    img = cv2.imread("D:\\Github\\ml\\ImageFiltering\\st52.png", cv2.IMREAD_COLOR)
    cv2.imshow("img", img)

    hsv= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_imgs = split_by_color(hsv)
    bin_imgs = get_bin_img(color_imgs)
    
    for color, bin_img in bin_imgs.items():
        #cv2.imshow(color, bin_img)
        #print(bin_img.shape)
        print("====" + color + "=====")
        cut_char(bin_img)

    
def cut_char(bin_img):
    # cut char
    x = 0
    while x < bin_img.shape[1]:
        for y in range(bin_img.shape[0]):
            if bin_img[y, x] != 0:
                char = bin_img[0:bin_img.shape[0], x-1:x+57]
                print(y, x, bin_img[y, x])
                cv2.imshow(str(x)+", "+str(y), char)
                x += 57
                break
        x += 1


    
    


def split_by_color(hsv):
    color_imgs = {}
    colors = {
        'red' : {
            'lower' : np.array([0, 50, 50]),
            'upper' : np.array([20, 255, 255])
        },
        'green' : {
            'lower' : np.array([50, 50, 50]),
            'upper' : np.array([70, 255, 255])
        },
        'blue' : {
            'lower' : np.array([110, 50, 50]),
            'upper' : np.array([130, 255, 255])
        }
    }

    for color, value in colors.items():
        mask = cv2.inRange(hsv, value['lower'], value['upper'])
        color_imgs[color] = cv2.bitwise_and(hsv, hsv, mask=mask)
    
    return color_imgs

def get_bin_img(color_imgs):
    bin_imgs = {}
    for color, color_img in color_imgs.items():
        # 흑백 binary img를 얻기 위해 일단 grayscale로 변환.
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        # grayscale img에 threshold를 적용해 binary img 얻음.
        dummy, bin_img = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
        bin_imgs[color] = bin_img
    
    return bin_imgs



    


if __name__ == "__main__":
    _main()

    k = cv2.waitKey(0)  

    if k == ord('s'):
        cv2.imwrite('D:\\Github\\ml\\ImageFiltering\\img.png', img)

    cv2.destroyAllWindows()
    

