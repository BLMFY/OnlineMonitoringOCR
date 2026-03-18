"""
相机管理模块
负责相机的打开、关闭、视频帧读取和处理
"""
import os
import cv2
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QImage

# 常见图像扩展名
_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp'}


class CameraManager(QObject):
    """相机管理类"""
    
    # 信号定义
    frame_ready = pyqtSignal(np.ndarray)  # 视频帧就绪信号
    camera_opened = pyqtSignal()  # 相机打开信号
    camera_closed = pyqtSignal()  # 相机关闭信号
    error_occurred = pyqtSignal(str)  # 错误信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap = None  # 视频捕获对象
        self.camera_id = 0  # 相机ID
        self.is_active = False  # 相机状态标志
        self.frame_width = 1280  # 默认宽度
        self.frame_height = 720  # 默认高度
        self._test_mode = False  # 是否测试模式
        self._test_image_frame = None  # 测试模式下图像文件的缓存帧
        
    def open_camera(self, camera_id=0, test_mode=False, test_material_path=""):
        """
        打开相机
        
        Args:
            camera_id: 相机ID，默认为0
            test_mode: 是否测试模式，使用测试素材替代真实相机
            test_material_path: 测试素材路径（图像或视频文件）
            
        Returns:
            bool: 成功返回True，失败返回False
        """
        try:
            # 如果相机已经打开，先关闭
            if self.cap is not None:
                self.close_camera()
            self._test_image_frame = None
            
            self._test_mode = test_mode and test_material_path and os.path.isfile(test_material_path)
            
            if self._test_mode:
                # 测试模式：打开图像或视频文件
                ext = os.path.splitext(test_material_path)[1].lower()
                if ext in _IMAGE_EXTENSIONS:
                    # 图像文件：读取一次并缓存
                    frame = cv2.imread(test_material_path)
                    if frame is None:
                        error_msg = f"无法读取图像文件: {test_material_path}"
                        self.error_occurred.emit(error_msg)
                        return False
                    self._test_image_frame = frame.copy()
                    self.cap = None  # 图像模式不需要 VideoCapture
                else:
                    # 视频文件
                    self.cap = cv2.VideoCapture(test_material_path)
                    if not self.cap.isOpened():
                        error_msg = f"无法打开视频文件: {test_material_path}"
                        self.error_occurred.emit(error_msg)
                        return False
                self.camera_id = -1  # 测试模式无相机ID
                self.is_active = True
                self.camera_opened.emit()
                print(f"[相机管理] 测试模式已开启: {test_material_path}")
                return True
            
            # 正常模式：打开真实相机
            self.camera_id = camera_id
            self.cap = cv2.VideoCapture(camera_id)
            
            if not self.cap.isOpened():
                error_msg = "无法打开相机，请检查相机连接"
                self.error_occurred.emit(error_msg)
                print(f"[错误] {error_msg}")
                return False
            
            # 设置相机分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            
            self.is_active = True
            self.camera_opened.emit()
            print(f"[相机管理] 相机 {camera_id} 已成功打开")
            return True
            
        except Exception as e:
            error_msg = f"打开相机时发生异常: {str(e)}"
            self.error_occurred.emit(error_msg)
            print(f"[错误] {error_msg}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return False
    
    def close_camera(self):
        """关闭相机"""
        self.is_active = False
        self._test_image_frame = None
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.camera_closed.emit()
        print("[相机管理] 相机已关闭")
    
    def read_frame(self):
        """
        读取视频帧
        
        Returns:
            tuple: (success, frame) 成功返回(True, frame)，失败返回(False, None)
        """
        # 测试模式：图像文件
        if self._test_image_frame is not None:
            self.frame_ready.emit(self._test_image_frame)
            return True, self._test_image_frame.copy()
        
        # 测试模式：视频文件
        if self._test_mode and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.frame_ready.emit(frame)
                return True, frame
            # 视频播放完毕，自动关闭
            self.close_camera()
            return False, None
        
        # 正常模式
        if self.cap is None or not self.cap.isOpened():
            return False, None
        
        ret, frame = self.cap.read()
        if ret:
            self.frame_ready.emit(frame)
        return ret, frame
    
    def is_opened(self):
        """
        检查相机是否打开
        
        Returns:
            bool: 相机打开返回True，否则返回False
        """
        if self._test_image_frame is not None:
            return True
        if self._test_mode and self.cap is not None and self.cap.isOpened():
            return True
        return self.cap is not None and self.cap.isOpened()
    
    def set_resolution(self, width, height):
        """
        设置相机分辨率
        
        Args:
            width: 宽度
            height: 高度
        """
        self.frame_width = width
        self.frame_height = height
        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    def set_zoom(self, value):
        """
        设置镜头变焦（占位实现）
        
        Args:
            value: 变焦值（0-100）
        """
        # TODO: 实现镜头变焦逻辑
        print(f"[相机管理] 设置镜头变焦: {value}")
    
    def frame_to_qimage(self, frame, target_size=None):
        """
        将OpenCV帧转换为QImage
        
        Args:
            frame: OpenCV BGR格式的帧
            target_size: 目标尺寸 (width, height)，如果为None则不缩放
            
        Returns:
            QImage: 转换后的QImage对象
        """
        if frame is None:
            return None
        
        # 将 BGR 格式转换为 RGB 格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 如果指定了目标尺寸，则缩放图像
        if target_size is not None:
            target_width, target_height = target_size
            if target_width > 0 and target_height > 0:
                # 保持宽高比缩放
                h, w = rgb_frame.shape[:2]
                frame_aspect = w / h
                target_aspect = target_width / target_height
                
                if frame_aspect > target_aspect:
                    # 宽度过大，以宽度为准
                    new_width = target_width
                    new_height = int(target_width / frame_aspect)
                else:
                    # 高度过大，以高度为准
                    new_height = target_height
                    new_width = int(target_height * frame_aspect)
                
                rgb_frame = cv2.resize(rgb_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # 转换为 QImage
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        return qt_image

