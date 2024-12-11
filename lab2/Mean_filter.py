import os

import cv2
import numpy as np


class MeanFilter(object):
    def __init__(self, img_path):
        self.originalImg = cv2.imread(img_path)
        self.greyImg = cv2.cvtColor(self.originalImg, cv2.COLOR_BGR2GRAY)


        if not os.path.exists('../result/meanFilter'):
            os.makedirs('../result/meanFilter')

    def ArithmeticMeanFilter(self, K_size = 3, filename='./result/meanFilter/arithmeticMeanFilter.jpg'):
        H = self.originalImg.shape[0]
        W = self.originalImg.shape[1]

        # zero padding
        pad = K_size // 2
        # 边界扩充半个滤波器大小
        out = np.zeros((H + pad *  2, W + pad * 2), dtype = np.float32)
        out[pad: pad + H, pad: pad + W] = self.greyImg.copy().astype(np.float32)
        tmp = out.copy()

        # filtering
        for y in range(H):
            for x in range(W):
                    out[pad + y, pad + x] = np.mean(tmp[y : y + K_size, x : x + K_size])

        # 去掉填充部分，返回最终的输出图像
        out = out[pad: pad + H, pad: pad + W].astype(np.int32)

        cv2.imwrite(filename, out)

    def GeometricMeanFilter(self, K_size=3, filename='./result/meanFilter/geometricMeanFilter.jpg'):
        H, W = self.originalImg.shape[:2]

        # 计算零填充大小
        pad = K_size // 2

        # 边界扩充
        out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float32)
        out[pad:pad + H, pad:pad + W] = self.greyImg.astype(np.float32)
        tmp = out.copy()

        # 进行滤波
        for y in range(H):
            for x in range(W):
                window = tmp[y:y + K_size, x:x + K_size] + 1e-10  # 加小常数避免出现负无穷
                out[pad + y, pad + x] = np.exp(np.sum(np.log(window))) ** (1 / (K_size * K_size))

        # 去掉填充部分
        out = out[pad:pad + H, pad:pad + W].astype(np.int32)

        # 保存结果
        cv2.imwrite(filename, out)

    def MedianFilter(self, K_size = 3, filename='./result/meanFilter/MedianFilter.jpg'):
        H = self.originalImg.shape[0]
        W = self.originalImg.shape[1]

        # zero padding
        pad = K_size // 2
        # 边界扩充半个滤波器大小
        out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float32)
        out[pad: pad + H, pad: pad + W] = self.greyImg.copy().astype(np.float32)
        tmp = out.copy()

        # filtering
        for y in range(H):
            for x in range(W):
                    out[pad + y, pad + x] = np.median(tmp[y: y + K_size, x: x + K_size])

        # 去掉填充部分，返回最终的输出图像
        out = out[pad: pad + H, pad: pad + W].astype(np.int32)

        cv2.imwrite(filename, out)

    # ω 是一个常数（通常在0到1之间），用于控制滤波的强度。
    # k 是一个参数，决定了滤波器的特性。不同的 k 值会影响到对噪点的抑制效果。
    def InverseHarmonicMeanFilter(self, K_size = 3, Q = 0, filename='./result/meanFilter/inverseHarmonicMeanFilter.jpg'):
        H = self.originalImg.shape[0]
        W = self.originalImg.shape[1]

        # zero padding
        pad = K_size // 2
        # 边界扩充半个滤波器大小
        out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float32)
        out[pad: pad + H, pad: pad + W] = self.greyImg.copy().astype(np.float32)
        tmp = out.copy()

        # filtering
        for y in range(H):
            for x in range(W):
                sum1 = 0
                sum2 = 0
                for b in range(y, y + K_size):
                    for a in range(x, x + K_size):
                        if tmp[b, a]:
                            sum1 += pow(tmp[b, a], Q + 1)
                            sum2 += pow(tmp[b, a], Q)
                if sum2 == 0:
                    out[pad + y, pad + x] = tmp[pad + y, pad + x]
                else:
                    out[pad + y, pad + x] = sum1 / sum2

        # 去掉填充部分，返回最终的输出图像
        out = out[pad: pad + H, pad: pad + W].astype(np.int32)

        cv2.imwrite(filename, out)

    def ModifiedAlphaMeanFilter(self, K_size = 3, d = 1, filename='./result/meanFilter/ModifiedAlphaMeanFilter.jpg'):
        H = self.originalImg.shape[0]
        W = self.originalImg.shape[1]

        # zero padding
        pad = K_size // 2
        # 边界扩充半个滤波器大小
        out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float32)
        out[pad: pad + H, pad: pad + W] = self.greyImg.copy().astype(np.float32)
        tmp = out.copy()

        # filtering
        for y in range(H):
            for x in range(W):
                adjFill = tmp[y: y + K_size, x: x + K_size]
                padSort = np.sort(adjFill.flatten())

                sumAlpha = np.sum(padSort[d:K_size * K_size - d - 1]) # 删除 d 个最大灰度值, d 个最小灰度值
                out[pad + y, pad + x] = sumAlpha / (K_size * K_size - 2 * d)  # 对剩余像素进行算术平均

        # 去掉填充部分，返回最终的输出图像
        out = out[pad: pad + H, pad: pad + W].astype(np.int32)

        cv2.imwrite(filename, out)

        # self.ShowPicture(self.ModifiedAlphaMeanFilter.__name__, out)

    def ShowPicture(self, method_name, out):
        # 拼接图片并显示
        combined_image = np.hstack((out, self.greyImg))
        combined_image = combined_image.astype(np.uint8)
        cv2.imshow(f'{method_name} - Combined Image', combined_image)
        # 等待按键事件，直到按下任意键
        cv2.waitKey(0)
        # 关闭所有打开的窗口
        cv2.destroyAllWindows()