import cv2
import easygui
import numpy as np
from scipy.interpolate import UnivariateSpline
 
def create_LUT_8UC1(x, y):
    spl = UnivariateSpline(x, y)
    return spl(range(256))

incr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256],
    [0, 70, 140, 210, 256])
decr_ch_lut = create_LUT_8UC1([0, 64, 128, 192, 256],
    [0, 30, 80, 120, 192])

def display_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def crop_minAreaRect(img, rect):
    # rotate img
    angle = rect[2]
    rows, cols = img.shape[0], img.shape[1]
    
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 0, 1) if angle<80 else cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    img_rot = cv2.warpAffine(img, M, (cols, rows))
    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0) 
    box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0
    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1], 
                       pts[1][0]:pts[2][0]]
    
    return img_crop


def main():
    
    file = easygui.fileopenbox(msg="Select your Image", title="Select an Image File",default="*.jpg",filetypes=["*.png","*.jpeg"])
    image = cv2.imread(file)
    x, y, _ = image.shape

    # Scaling down by a factor of 5
    image = cv2.resize(image, (x//2, y//2))

    # Everything is better with binary
    greyyed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Applying a gaussian blur to filter out excess redundent edges
    blurred = cv2.GaussianBlur(greyyed, (17, 17), 0)
    # Using canny because we can, hey why not its simple :-P
    edges = cv2.Canny(blurred, threshold1=10, threshold2=100, apertureSize=3)

    display_image(edges)

    # We will have to find the countors
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(contours[0])
    image = crop_minAreaRect(image, rect)

    # ------------FOR ENHANCED CONTRAST OF OUR NEGETIVE PHOTOS---------------
    image = cv2.bitwise_not(image)

    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab = cv2.merge((l2, a, b))  # merge channels
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # -----------Adding a warm effect-----------
    # ref-> http://www.askaswiss.com/2016/02/how-to-manipulate-color-temperature-opencv-python.html
    c_b, c_g, c_r = cv2.split(image)
    c_r = cv2.LUT(c_r, incr_ch_lut).astype(np.uint8)
    c_b = cv2.LUT(c_b, decr_ch_lut).astype(np.uint8)
    image = cv2.merge((c_b, c_g, c_r))




    display_image(image)

    cv2.imwrite("rectified.JPG", image)
    
if __name__ == '__main__':
    main()