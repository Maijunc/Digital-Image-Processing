import cv2
import lab6.color_image_transform as CIT
import numpy as np


image = cv2.imread('src/img.png', cv2.IMREAD_COLOR)
# 转换为 HSV 空间
hsi_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)


cit = CIT.ColorImageTransform()

kernel_size = 5
sigma = 1.0
gaussian_k = cit.gaussian_kernel(kernel_size, sigma)
# 定义拉普拉斯算子
laplacian_kernel = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]], dtype=np.float32)

rgb_blurred_image = cit.apply_filter(image, gaussian_k)
hsi_blurred_image = cit.apply_filter(hsi_image, gaussian_k)

alpha = 1.0  # 控制锐化程度的参数
rgb_sharpened_image = cit.apply_filter(rgb_blurred_image, laplacian_kernel)
hsi_sharpened_image = cit.apply_filter(hsi_blurred_image, laplacian_kernel)
# 将拉普拉斯结果加回到模糊图像中 反向叠加，使边缘更突出
rgb_sharpened_image = cv2.addWeighted(rgb_blurred_image, 1, rgb_sharpened_image, -alpha, 0)
rgb_sharpened_image = np.clip(rgb_sharpened_image, 0, 255).astype(np.uint8)

hsi_sharpened_image = cv2.addWeighted(hsi_blurred_image, 1, hsi_sharpened_image, -alpha, 0)
hsi_sharpened_image = np.clip(hsi_sharpened_image, 0, 255).astype(np.uint8)

cv2.imshow('Original Image', image)
cv2.imshow('Gaussian blurred Image', rgb_blurred_image.astype(np.uint8))
cv2.imshow('Sharpened and blurred Image', rgb_sharpened_image)

cv2.imshow('HSI Original Image', hsi_image)
cv2.imshow('HSI Gaussian blurred Image', hsi_blurred_image.astype(np.uint8))
cv2.imshow('HSI Sharpened and blurred Image', hsi_sharpened_image)

# 等待按键再关闭
cv2.waitKey(0)
cv2.destroyAllWindows()