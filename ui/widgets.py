"""
自定义UI控件
"""
from typing import List, Optional, Tuple

from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtCore import Qt, QSize, QPointF, QRectF
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QPolygonF, QPainterPath


class AspectRatioLabel(QLabel):
    """保持固定宽高比的标签控件，支持覆盖层绘制（虚线框、提示点、检测区域）"""
    
    def __init__(self, aspect_ratio=16/9, parent=None):
        super().__init__(parent)
        self.aspect_ratio = aspect_ratio
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setScaledContents(True)
        # 覆盖层数据（label 坐标系）
        self._hint_rect_start: Optional[Tuple[float, float]] = None
        self._hint_rect_end: Optional[Tuple[float, float]] = None
        self._hint_points: List[Tuple[float, float]] = []
        self._target_points: List[Tuple[float, float]] = []
        self._target_preview: Optional[Tuple[float, float]] = None  # 框选时未松开鼠标的预览点
        self._regions: List[List[Tuple[float, float]]] = []  # 每个区域 4 个顶点
    
    def set_hint_rect(self, start: Optional[Tuple[float, float]], end: Optional[Tuple[float, float]]):
        """设置区域提示的矩形框（label 坐标）"""
        self._hint_rect_start = start
        self._hint_rect_end = end
        self.update()
    
    def set_hint_points(self, points: List[Tuple[float, float]]):
        """设置坐标提示点（label 坐标）"""
        self._hint_points = list(points)
        self.update()

    def set_target_points(self, points: List[Tuple[float, float]], preview: Optional[Tuple[float, float]] = None):
        """设置框选目标的点（label 坐标），preview 为未松开鼠标时的预览点"""
        self._target_points = list(points)
        self._target_preview = preview
        self.update()
    
    def set_regions(self, regions: List[List[Tuple[float, float]]]):
        """设置检测到的文本区域多边形（label 坐标，每个区域 4 个顶点）"""
        self._regions = [list(poly) for poly in regions]
        self.update()
    
    def clear_overlay(self):
        """清除所有覆盖层"""
        self._hint_rect_start = None
        self._hint_rect_end = None
        self._hint_points = []
        self._target_points = []
        self._target_preview = None
        self._regions = []
        self.update()
    
    def paintEvent(self, event):
        """先绘制父类内容（含 pixmap），再绘制覆盖层"""
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # 红色虚线矩形（区域提示拖拽中）
        if self._hint_rect_start and self._hint_rect_end:
            x1, y1 = self._hint_rect_start
            x2, y2 = self._hint_rect_end
            rect = QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(rect)
        
        # 红色星星（坐标提示点）
        for px, py in self._hint_points:
            self._draw_star(painter, px, py, 8)

        # 绿色圆点 + 连线（框选目标点）
        if self._target_points or self._target_preview:
            pen = QPen(QColor(0, 200, 0), 2)
            brush = QBrush(QColor(0, 255, 0))
            painter.setPen(pen)
            painter.setBrush(brush)
            # 画已确定的点
            for px, py in self._target_points:
                painter.drawEllipse(QPointF(px, py), 4, 4)
            # 画已确定的点之间的连线
            if len(self._target_points) >= 2:
                painter.setBrush(Qt.NoBrush)
                for i in range(len(self._target_points) - 1):
                    p1 = self._target_points[i]
                    p2 = self._target_points[i + 1]
                    painter.drawLine(QPointF(p1[0], p1[1]), QPointF(p2[0], p2[1]))
            # 画预览点及与上一点的连线（鼠标未松开时）
            if self._target_preview:
                px, py = self._target_preview
                if self._target_points:
                    last = self._target_points[-1]
                    painter.drawLine(QPointF(last[0], last[1]), QPointF(px, py))
                # 预览点用空心圆表示
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QPointF(px, py), 4, 4)
        
        # 淡绿色半透明四边形（检测到的文本区域）
        for poly in self._regions:
            if len(poly) >= 3:
                qpoly = QPolygonF([QPointF(p[0], p[1]) for p in poly])
                painter.setPen(QPen(QColor(0, 200, 0), 2))
                painter.setBrush(QBrush(QColor(0, 255, 0, 80)))
                painter.drawPolygon(qpoly)
        
        painter.end()
    
    def _draw_star(self, painter: QPainter, cx: float, cy: float, r: float):
        """在 (cx, cy) 绘制红色五角星，外径 r"""
        import math
        path = QPainterPath()
        for i in range(5):
            angle = -90 + i * 72  # 从顶部开始
            rad = math.radians(angle)
            x = cx + r * math.cos(rad)
            y = cy + r * math.sin(rad)
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
            # 内角
            inner_angle = angle + 36
            inner_rad = math.radians(inner_angle)
            ix = cx + r * 0.4 * math.cos(inner_rad)
            iy = cy + r * 0.4 * math.sin(inner_rad)
            path.lineTo(ix, iy)
        path.closeSubpath()
        painter.setPen(QPen(QColor(255, 0, 0), 2))
        painter.setBrush(QBrush(QColor(255, 0, 0)))
        painter.drawPath(path)
        
    def sizeHint(self):
        """返回建议的尺寸，保持宽高比"""
        if self.parent() is None:
            return super().sizeHint()
            
        # 获取父容器的可用空间
        parent_size = self.parent().size()
        parent_width = max(100, parent_size.width() - 20)  # 减去边距，最小100
        parent_height = max(100, parent_size.height() - 20)
        
        # 根据宽高比计算合适的尺寸
        if parent_width / parent_height > self.aspect_ratio:
            # 宽度过大，以高度为准
            new_height = parent_height
            new_width = int(new_height * self.aspect_ratio)
        else:
            # 高度过大，以宽度为准
            new_width = parent_width
            new_height = int(new_width / self.aspect_ratio)
        
        return QSize(new_width, new_height)
    
    def minimumSizeHint(self):
        """返回最小尺寸"""
        return QSize(320, int(320 / self.aspect_ratio))  # 最小 16:9 比例，宽度320
    
    def hasHeightForWidth(self):
        """告诉布局系统这个控件的高度依赖于宽度"""
        return True
    
    def heightForWidth(self, width):
        """根据宽度返回对应的高度，保持宽高比"""
        return int(width / self.aspect_ratio)

