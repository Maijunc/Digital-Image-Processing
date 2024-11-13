import cv2
import numpy as np
from math import exp

class LaplaceTranslation(object):
    def BasicLaplaceTranslation(self, srcImage):
        if len(srcImage.shape) != 2:
            srcImage = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
        H = srcImage.shape[0]
        W = srcImage.shape[1]

        border = 1
        out = srcImage.copy().astype(np.float32)
        tmp = out.copy()
        for i in range(border, H - border):
            for j in range(border, W - border):
                out[i, j] = 9 * tmp[i, j] - tmp[i - 1, j - 1] - tmp[i - 1, j] - tmp[i - 1, j + 1] \
                - tmp[i, j - 1] - tmp[i, j + 1] \
                - tmp[i + 1, j - 1] - tmp[i + 1, j] - tmp[i + 1, j + 1] \

        return out

    def GaussianFilter(self, srcImage, ksize, sigma):
        if len(srcImage.shape) != 2:
            srcImage = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
        # pi = 3.1415926
        templateMartix = np.zeros((ksize, ksize), dtype=np.float32)
        # 向下取整
        origin = ksize // 2
        sum = 0
        for i in range(ksize):
            x2 = pow(i - origin, 2)
            for j in range(ksize):
                y2 = pow(j - origin, 2)
                g = exp(-(x2 + y2) / (2 * sigma * sigma))
                # 高斯函数前的常数可以不用计算，会在归一化的过程中给消去
                # g /= 2 * pi * sigma;
                sum += g
                templateMartix[i, j] = g

        for i in range(ksize):
            for j in range(ksize):
                templateMartix[i, j] /= sum
                print(templateMartix[i, j], end = ' ')
            print('\n')

        H = srcImage.shape[0]
        W = srcImage.shape[1]

        # zero padding
        pad = ksize // 2
        # 边界扩充半个滤波器大小
        out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float32)
        out[pad: pad + H, pad: pad + W] = srcImage.copy().astype(np.float32)
        tmp = out.copy()

        # filtering
        for i in range(pad, H + pad):
            for j in range(pad, W + pad):
                sum_f = np.float32(0)
                for a in range(-pad, pad):
                    for b in range(-pad, pad):
                        sum_f += templateMartix[a, b] * tmp[i + a, j + b]
                if sum_f < 0:
                    sum_f = 0
                elif sum_f > 255:
                    sum_f = 255
                out[i, j] = sum_f


        # 去掉填充部分
        out = out[pad: pad + H, pad: pad + W].astype(np.int32)

        return out

    def GaussianLaplaceTranslation(self, srcImage, ksize, sigma):
        return self.BasicLaplaceTranslation(self.GaussianFilter(srcImage, ksize, sigma))

    def CV2GaussianLaplaceTranslation(self, srcImage, ksize, sigma):
        return self.BasicLaplaceTranslation(cv2.GaussianBlur\
                                                (cv2.cvtColor\
                                                     (srcImage, cv2.COLOR_BGR2GRAY), (ksize, ksize), sigma))
