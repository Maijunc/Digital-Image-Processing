import os
import cv2
import numpy as np


class TransformEnhance(object):

    def __init__(self, img_path, file_path='./result/enhance'):
        self.originalImg = cv2.imread(img_path)
        if self.originalImg is None:
            raise ValueError(f"Image not found at the path: {img_path}")

        self.greyImg = cv2.cvtColor(self.originalImg, cv2.COLOR_BGR2GRAY)
        self.file_path = file_path

        if not os.path.exists(file_path):
            os.makedirs(file_path)

    def LogTransform(self, c=1):
        # 使用更大的 c 值
        out = c * np.log1p(self.greyImg.astype(np.float32))  # 转换为 float32

        # 确保输出在 [0, 255] 范围内
        out = np.clip(out, 0, 255)

        out = out.astype(np.uint8)

        # 存储图片
        cv2.imwrite(os.path.join(self.file_path, 'logTransform.jpg'), out)

    # s = c * r ^ v
    def PowerTransform(self, c, v):
        H = self.greyImg.shape[0]
        W = self.greyImg.shape[1]

        out = np.zeros((H, W), dtype = np.float32)

        for i in range(H):
            for j in range(W):
                out[i][j] = c * pow(self.greyImg[i][j], v)

        out = out.astype(np.uint8)

        # 存储图片
        cv2.imwrite(os.path.join(self.file_path, 'powerTransfrom.jpg'), out)

