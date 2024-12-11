import cv2
import numpy as np

class EdgeDetection(object):
    # 通过传参不同来适配不同的边缘检测算子
    def OneDimensionDetection(self, srcImg, kernelx, kernely):
        # 灰度化处理图像
        greyImage = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)

        # Roberts算子
        # kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
        # kernely = np.array([[0, -1], [1, 0]], dtype=int)

        # Sobel算子
        # kernelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=int)
        # kernely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=int)

        # Prewitt算子
        # kernelx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
        # kernely = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=int)

        # zero padding
        pad = kernelx.shape[0] // 2
        H = greyImage.shape[0]
        W = greyImage.shape[1]

        # 扩充边缘
        out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float32)
        out[pad: pad + H, pad: pad + W] = greyImage.copy().astype(np.float32)
        tmp = out.copy()

        # filtering
        for i in range(pad, H + pad):
            for j in range(pad, W + pad):
                G_X = np.float32(0)
                for h in range(kernelx.shape[0]):
                    for w in range(kernelx.shape[1]):
                        G_X += tmp[i + h, j + w] * kernelx[h, w]
                        # print(G_X, end=' ')
                # print()
                G_Y = np.float32(0)
                for h in range(kernely.shape[0]):
                    for w in range(kernely.shape[1]):
                        G_Y += tmp[i + h, j + w] * kernely[h, w]
                out[i, j] = (abs(G_X) * 0.5 + abs(G_Y) * 0.5)

        # 去掉填充部分，返回最终的输出图像
        out = out[pad: pad + H, pad: pad + W].astype(np.uint8)

        # 使用opencv卷积函数
        x = cv2.filter2D(greyImage, cv2.CV_16S, kernelx)
        y = cv2.filter2D(greyImage, cv2.CV_16S, kernely)
        # 转uint8
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        cvRes = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)


        return out,cvRes

    def Laplace(self, srcImg, kernel):
        # 灰度化处理图像
        greyImage = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)
        Laplaces = cv2.filter2D(greyImage, cv2.CV_16S, kernel)

        return cv2.convertScaleAbs(Laplaces)