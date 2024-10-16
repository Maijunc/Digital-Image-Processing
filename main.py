import cv2
import numpy as np
import Histogram_equalization as HE

if __name__ == '__main__':
    img_path = ''
    img = cv2.imread(img_path, 0)
    equ = cv2.equalizeHist(img)
    res = np.hstack((img, equ))  # stacking images side-by-side
    cv2.imwrite('./result/opencvEqualization.jpg', res)

    he = HE.HistogramEqualization(img_path)
    he.draw_histogram()
    he.draw_cd_f()
    he.equalization()
    he.draw_new_histogram()