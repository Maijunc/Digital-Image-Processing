import numpy as np
import cv2

class ColorImageTransform(object):
    def gaussian_kernel(self, size, sigma):
        """生成高斯核"""
        kernel = np.zeros((size, size))
        mean = size // 2

        # 生成高斯核
        for x in range(size):
            for y in range(size):
                exponent = (x - mean) ** 2 + (y - mean) ** 2
                kernel[x, y] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-exponent / (2 * sigma ** 2))

        return kernel / np.sum(kernel)  # 归一化


    def apply_filter(self, image, kernel):
        """应用卷积"""
        kernel_size = kernel.shape[0]
        pad = kernel_size // 2

        # 添加边界填充
        padded_image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        filtered_image = np.zeros_like(image, dtype=np.float32)

        # 卷积运算
        for i in range(pad, padded_image.shape[0] - pad):
            for j in range(pad, padded_image.shape[1] - pad):
                for c in range(padded_image.shape[2]):
                    # 对应区域与高斯核相乘并求和
                    region = padded_image[i - pad:i + pad + 1, j - pad:j + pad + 1, c]
                    filtered_image[i - pad, j - pad, c] = np.sum(region * kernel)

        return filtered_image