import cv2
import numpy as np

class TransformEnhance(object):
    def otsu_thresholding(self, image):
        # 计算直方图  用_来占位忽略掉第二个返回值
        histogram, _ = np.histogram(image, bins=256, range=(0, 256))

        # 计算总像素数
        total_pixels = image.shape[0] * image.shape[1]

        # 变量初始化
        current_max, threshold = 0, 0
        sum_total, sum_foreground = 0, 0

        # 计算总灰度值
        for i in range(256):
            sum_total += i * histogram[i]

        # 背景的权重 和 前景的权重，权重：在当前阈值以下或以上（即被视为背景）的所有像素的总数。
        weight_background, weight_foreground = 0, 0

        # 开始计算
        for t in range(256):
            weight_background += histogram[t] # 背景的权重

            if weight_background == 0:
                continue

            weight_foreground = total_pixels - weight_background # 前景的权重

            # 如果前景权重为零，表示没有像素被视为前景
            if weight_foreground == 0:
                break

            sum_foreground += t * histogram[t] # 前景的灰度总和

            # 后景的灰度均值
            mean_background = sum_foreground / weight_background
            # 前景的灰度均值
            mean_foreground = (sum_total - sum_foreground) / weight_foreground

            # ω0 = N0 / M×N(1)
            # ω1 = N1 / M×N(2)
            # N0 + N1 = M×N(3)
            # ω0 + ω1 = 1(4)
            # μ = ω0 * μ0 + ω1 * μ1(5)
            # g = ω0(μ0 - μ) ^ 2 + ω1(μ1 - μ) ^ 2(6) 类间方差

            # 将式(5)
            # 代入式(6), 得到等价公式:
            # g = ω0 * ω1(μ0 - μ1) ^ 2(7)

            # 这里的w0和w1不需要确实求出来，直接使用像素点的总值也可以表示权重，最终都是用于比较的

            # 类间方差
            between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

            # 最大化类间方差
            if between_class_variance > current_max:
                current_max = between_class_variance
                threshold = t

        return threshold

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
        filtered_image = np.zeros_like(image)

        # 卷积运算
        for i in range(pad, padded_image.shape[0] - pad):
            for j in range(pad, padded_image.shape[1] - pad):
                # 对应区域与高斯核相乘并求和
                region = padded_image[i - pad:i + pad + 1, j - pad:j + pad + 1]
                filtered_image[i - pad, j - pad] = np.sum(region * kernel)

        return filtered_image