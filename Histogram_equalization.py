import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
import os

class HistogramEqualization(object):
    def __init__(self, img_file):
        # 读取图像

        self.OriginalImg = cv2.imread(img_file)
        self.Histogram = np.zeros((1, 256), dtype=np.int32)
        self.NewHistogram = np.zeros((1, 256), dtype=np.int32)
        self.Cumulative_distribution = np.zeros((1, 256), dtype=np.float32)
        # 用于后续直方图均衡化计算使用
        self.cdfmin = sys.maxsize
        self.cdfmax = -sys.maxsize - 1

        # 设置Matplotlib绘制图形所使用的字体
        font_path = '/Library/Fonts/Arial Unicode.ttf'  # macOS
        font_prop = fm.FontProperties(fname=font_path)
        matplotlib.rcParams['font.family'] = font_prop.get_name()
        matplotlib.rcParams['axes.unicode_minus'] = False

        if not os.path.exists('./result/'):
            os.makedirs('./result/')

    def resize(self, size: tuple = (64, 64)):
        self.OriginalImg = cv2.resize(self.OriginalImg, size)

    def draw_histogram(self):
        self.Histogram[0] = cv2.calcHist([self.OriginalImg], [0], None, [256], [0, 256]).flatten()

        # 绘制直方图
        plt.figure(figsize=(10, 5))
        plt.xlim(0, 255)
        plt.xlabel('灰度值')
        plt.ylabel('频数')
        plt.plot(self.Histogram[0])
        plt.title('原始图片直方图')
        plt.savefig('./result/origin_histogram.jpg')

        # 存储原始图片
        cv2.imwrite('./result/origin_img.jpg', self.OriginalImg)

    def draw_cd_f(self):
        # 累加直方图，得到累计分布函数，并绘制图像
        height = self.OriginalImg.shape[0]
        width = self.OriginalImg.shape[1]
        N = height * width  # 所有图像点个数
        for i in range(256):
            if i == 0:
                self.Cumulative_distribution[0][i] = self.Histogram[0][i] / N
            elif i == 255:
                self.Cumulative_distribution[0][i] = 1.0
            else:
                self.Cumulative_distribution[0][i] = self.Histogram[0][i] / N + self.Cumulative_distribution[0][i - 1]
            self.cdfmin = min(self.cdfmin, self.Cumulative_distribution[0][i])
            self.cdfmax = max(self.cdfmax, self.Cumulative_distribution[0][i])

        # 绘制图像
        plt.figure()
        plt.xlim(0, 255)
        plt.ylim(0, 1)
        plt.xlabel('灰度值')
        plt.ylabel('概率')
        plt.title('累积分布函数图像')
        plt.plot(range(256), self.Cumulative_distribution[0, :], color='green', linewidth=0.5)
        plt.savefig('./result/cdf.jpg')

    def equalization(self, filename='./result/new_img.jpg'):
        """
        利用计算得到的累积分布函数对原始图像像素进行均衡化，得到映射函数
        :return: None
        """
        # 累积分布函数计算完成后，进行I和c的缩放，把值域缩放到0~255的范围之内
        self.Cumulative_distribution = self.Cumulative_distribution * 255
        self.cdfmin *= 255
        self.cdfmax *= 255
        height = self.OriginalImg.shape[0]
        width = self.OriginalImg.shape[1]
        f = np.zeros((1, 256), dtype=np.int32)  # 映射函数
        for i in range(256):
            f[0][i] = (self.Cumulative_distribution[0][i] - self.cdfmin) / (self.cdfmax - self.cdfmin) * 255
        f = f.astype(np.int32)

        # f为得到的映射，据此生成新的图像
        self.NewImg = np.zeros((self.OriginalImg.shape))
        self.NewImg = self.OriginalImg.copy()
        for row in range(height):
            for col in range(width):
                self.NewImg[row][col] = f[0][self.NewImg[row][col]]
        cv2.imwrite(filename, self.NewImg)

    def draw_new_histogram(self, filename='./result/new_histogram.jpg'):
        """绘制新图片的直方图"""
        # 计算像素点，得到原始图片直方图
        height = self.OriginalImg.shape[0]
        width = self.OriginalImg.shape[1]

        self.NewHistogram[0] = cv2.calcHist([self.NewImg], [0], None, [256], [0, 256]).flatten()

        # 绘制直方图
        plt.figure(figsize=(10, 5))
        plt.xlim(0, 255)
        plt.xlabel('灰度值')
        plt.ylabel('频数')
        plt.plot(self.NewHistogram[0])
        plt.title('均衡化后的图片直方图')
        plt.savefig('./result/new_histogram.jpg')