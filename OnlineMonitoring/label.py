from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtCore import QRect, Qt
# from demo.label import MyLabel

def overlap(box1, box2): # 判断框体相交函数
    minx1, miny1, maxx1, maxy1 = box1
    minx2, miny2, maxx2, maxy2 = box2
    minx = max(minx1, minx2)
    miny = max(miny1, miny2)
    maxx = min(maxx1, maxx2)
    maxy = min(maxy1, maxy2)
    if minx >= maxx or miny >= maxy:
        return False
    else:
        return True

def isPointInRect(x, y, rect):
    a = (rect[2] - rect[0])*(y - rect[1]) - (rect[3] - rect[1])*(x - rect[0])
    b = (rect[4] - rect[2])*(y - rect[3]) - (rect[5] - rect[3])*(x - rect[2])
    c = (rect[6] - rect[4])*(y - rect[5]) - (rect[7] - rect[5])*(x - rect[4])
    d = (rect[0] - rect[6])*(y - rect[7]) - (rect[1] - rect[7])*(x - rect[6])
    if ((a>0)&(b>0)&(c>0)&(d>0))or((a<0)&(b<0)&(c<0)&(d<0)):
        return True
    else:
        return False

class MyLabel(QtWidgets.QLabel):
    x0 = 0
    y0 = 0
    x1 = 0
    y1 = 0
    rec_box = []
    rec_area = []
    edit_box = None
    edit_id = None
    key = 0
    '''key=1 画框
    key=2 绘制或删除完毕，框体可选择，识别可开始
    key=3 已经选中框体，标记为蓝色，可删除
    key=4 绘制完毕后可开始的开关
    key=5 开始点选四点区域'''
    flag_paint = False # 绘制开关
    flag_line = False
    flag_check = False
    #鼠标点击事件
    def mousePressEvent(self,event):
        if self.key == 1: # 
        #     self.flag = True        
            self.x0 = event.x()
            self.y0 = event.y()
        elif self.key == 2:
            for i, area in enumerate(self.rec_box):
                if len(area) == 4:
                    if (event.x() >= area[0])&(event.y() >= area[1])&(event.x() <= area[2])&(event.y() <= area[3]):
                        self.edit_box = area
                        self.edit_id = i+1
                        self.key = 3
                        break
                elif len(area) == 8:
                    if isPointInRect(event.x(), event.y(), area):
                        self.edit_box = area
                        self.edit_id = i+1
                        self.key = 6
                        break
                # else:
                self.edit_box = None
                self.edit_id = None
                self.key = 4
        elif self.key == 5:
             self.flag_check = True
             self.flag_line = True
             self.x1 = event.x()
             self.y1 = event.y()
             
    #鼠标释放事件
    def mouseReleaseEvent(self,event):
        if self.key == 1:
        #     self.flag = False
            box = [min(self.x0,self.x1),
                   min(self.y0,self.y1),
                   max(self.x0,self.x1),
                   max(self.y0,self.y1)]
            self.rec_box.append(box)
            if len(self.rec_box) > 1:
                for i in range(len(self.rec_box)-1):
                    if len(self.rec_box[i]) == 4:   
                        if overlap(box, self.rec_box[i]):
                            self.rec_box.remove(box)
                            break 
            self.setCursor(Qt.ArrowCursor)
            self.flag_paint = False
            self.key = 4
        if self.key == 5:
            self.flag_check = False
            self.rec_area.append(self.x1)
            self.rec_area.append(self.y1)
            self.x0 = self.x1
            self.y0 = self.y1
            if len(self.rec_area) >= 8:
                self.rec_box.append(self.rec_area)
                self.rec_area = []
                self.flag_line = False
                self.key = 4
                self.setCursor(Qt.ArrowCursor)
    #鼠标移动事件
    def mouseMoveEvent(self,event):
        if self.key == 1:
            self.flag_paint = True
        #     if self.flag:
            self.x1 = event.x()
            self.y1 = event.y()
            if (self.x1 - self.x0) > 6 *(self.y1 - self.y0):
                self.x1 = self.x0 + 6 *(self.y1 - self.y0)
            self.update()
        if self.key == 5:
            
            self.x1 = event.x()
            self.y1 = event.y()
            self.update()
    #绘制事件
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        if self.flag_paint:
            rect =QRect(min(self.x0,self.x1),min(self.y0,self.y1), abs(self.x1-self.x0), abs(self.y1-self.y0))
            painter.setPen(QPen(Qt.red,2,Qt.SolidLine))
            painter.drawRect(rect)
        if self.flag_line:
            painter.setPen(QPen(Qt.red,2,Qt.SolidLine))
            if len(self.rec_area) >= 2:
                painter.drawLine(self.x0, self.y0, self.x1, self.y1)
            if (len(self.rec_area) >= 6) and self.flag_check:
                painter.drawLine(self.rec_area[0], self.rec_area[1], self.x1, self.y1)
            else:
                painter.drawPoint(self.x1, self.y1)

# python -m PyQt5.uic.pyuic -x demo/dialog_ex.ui -o demo/dialog1.py