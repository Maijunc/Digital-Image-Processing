import cv2
import numpy as np
import lab5.Threshold_partition as TP

# 加载图像
image = cv2.imread('src/实验5-3.tif', cv2.IMREAD_GRAYSCALE)
tp = TP.TransformEnhance()

# 获取迭代阀值
iteration_threshold = tp.iteration_thresholding(image)
_, segmented_image = cv2.threshold(image, iteration_threshold, 255, cv2.THRESH_BINARY)
cv2.imshow('Original Image', image)
cv2.imshow('Iteration Segmented Image', segmented_image)

''' Otsu阈值图像分割

kernel_size = 5
sigma = 1.0
gaussian_k = tp.gaussian_kernel(kernel_size, sigma)
blurred_image_mine = tp.apply_filter(image, gaussian_k)
blurred_image_cv2 = cv2.blur(image, (3, 3), sigma)

# 获取Otsu阈值
otsu_threshold_mine = tp.otsu_thresholding(blurred_image_mine)
otsu_threshold_cv2 = tp.otsu_thresholding(blurred_image_cv2)
otsu_threshold_without_blur = tp.otsu_thresholding(image)
# 应用阈值
_, segmented_image_mine = cv2.threshold(blurred_image_mine, otsu_threshold_mine, 255, cv2.THRESH_BINARY)
_, segmented_image_cv2 = cv2.threshold(blurred_image_cv2, otsu_threshold_cv2, 255, cv2.THRESH_BINARY)
_, segmented_image_without_blur = cv2.threshold(image, otsu_threshold_without_blur, 255, cv2.THRESH_BINARY)

# 显示图像
# cv2.imshow('Original Image', image)
# cv2.imshow('Blurred Otsu Segmented Image CV2 ' + 'sigma = ' + str(sigma), segmented_image_cv2)
cv2.imshow('Blurred Image ' + 'sigma = ' + str(sigma), blurred_image_mine)
cv2.imshow('Blurred Otsu Segmented Image ' + 'sigma = ' + str(sigma), segmented_image_mine)
# cv2.imshow('Otsu Segmented Image', segmented_image_without_blur)

'''

# 等待按键再关闭
cv2.waitKey(0)
cv2.destroyAllWindows()

