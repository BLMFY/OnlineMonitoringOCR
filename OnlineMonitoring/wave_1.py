#计算示波器波形图片的占空比可以使用Python中的numpy和matplotlib库。首先，读取示波器波形图片，并将其转化为灰度图像。然后，将图像中的波形区域提取出来，并计算出波形的周期和高电平、低电平持续的时间。最终，根据占空比的定义，计算出占空比。

#以下是示波器波形图片占空比计算的Python代码示例：

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import math
    
def wave_detect(x1,y1,x2,y2,colorThresh,color,imgOri):
    imgGreen = imgOri[:,:,color] 
    ''' 
    imgGreen 为 绿色通道的 色彩强度图 (注意不是原图的灰度转换结果)
    绿色通道转换为二值图像，生成遮罩 Mask、逆遮罩 MaskInv
    如果背景不是绿屏而是其它颜色，可以采用对应的颜色通道进行阈值处理 (不宜基于灰度图像进行固定阈值处理，性能差异很大)
    colorThresh = 160 # 绿屏背景的颜色阈值 (注意研究阈值的影响) 
    '''
    ret, binary = cv2.threshold(imgGreen, colorThresh, 255, cv2.THRESH_BINARY_INV) # 转换为二值图像，生成遮罩，抠图区域黑色遮盖 
    m = (y2-y1)/(x2-x1)
    n = y2-m*x2
    list=[]
    rows, cols = binary.shape#获取图像像素的行数和列数
    thh = max(5, cols//16)
    i = 0
     
    while i < cols:
        j = int(m*i+n)
        if j < rows and j >0 :
            if binary[j][i] == 0 :

                list.append((i,j))

                i = i+thh
            i = i+1
        else:
            err = "画线坐标超出框选区域！"
            return("00.0",binary,err)     
       
    if len(list) >= 3:
        for p in list :
            cv2.circle(binary,p,3,(0,0,225),3)
        a= list[1][0] - list[0][0]
        b = list[1][1] - list[0][1]
        L1 = math.sqrt(a**2 + b**2)
        c = list[2][0] - list[1][0]
        d = list[2][1] - list[1][1]
        L2 = math.sqrt(c**2 + d**2)
        zkb = (L1/(L1+L2))*100
        zkb = str(zkb)
        zkb = zkb[:4]
        err = "检测正常!"
        #return("00.0",imgOri,err)
        
        return(zkb,binary,err)
    else:
        err = "未检测到交点，请调整!"
        return("00.0",binary,err)
    
def bxsx(image,color,thresh):

    imgGreen = image[:,:,color] # imgGreen 为 绿色通道的 色彩强度图 (注意不是原图的灰度转换结果)
    ret, binary = cv2.threshold(imgGreen, thresh, 255, cv2.THRESH_BINARY_INV) # 转换为二值图像，生成遮罩，抠图区域黑色遮盖

    return(binary)
     
    
    