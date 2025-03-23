import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import math
import copy


def pixel_equal(image1, image2, x, y,threshold):
    """
    判断两个像素是否相同
    :param image1: 图片1
    :param image2: 图片2
    :param x: 位置x
    :param y: 位置y
    :return: 像素是否相同
    
    # 取两个图片像素点
    piex1 = image1[x, y]
    piex2 = image2[x, y]
    print(piex1)
    print(piex2)
    threshold 
    # 比较每个像素点的RGB值是否在阈值范围内，若两张图片的RGB值都在某一阈值内，则我们认为它的像素点是一样的
    if abs(piex1[0] - piex2[0]) < threshold and abs(piex1[1]- piex2[1]) < threshold and abs(piex1[2] - piex2[2]) < threshold:
        return True
    else:
        return False
    """
    
    piex1 = int(image1[x, y])
    piex2 = int(image2[x, y])

    if abs(piex1 - piex2) < threshold:
        return True
    else:
        return False


def compare(self,image1,image2,limit_value):
    

    """
    进行比较
    :param image1:图片1
    :param image2: 图片2
    :return:
    """
    left = 1		# 坐标起始位置
    right_num = 0	# 记录相同像素点个数
    false_num = 0	# 记录不同像素点个数
    all_num = 0		# 记录所有像素点个数
    red_color = (255,0,0)
    image2_0 = copy.deepcopy(image2)

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    image3 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    rows, cols= image1.shape#获取图像像素的行数和列数
    

    
    for i in range(cols) :#x坐标系
        for j in range(rows) :#y坐标系
            
            if pixel_equal(image1, image3, j,i,limit_value):
                right_num += 1
            else:
                false_num += 1
                image2_0[j,i]= red_color
            all_num += 1
            
    same_rate = right_num / all_num	*100	# 相同像素点比例
    nosame_rate = false_num / all_num *100	# 不同像素点比例
    nosame_rate = round(nosame_rate, 2)

    return false_num, image2_0, nosame_rate