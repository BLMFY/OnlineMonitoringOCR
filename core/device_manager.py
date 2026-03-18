"""
设备管理模块
负责相机设备检测和硬件设备管理
"""
import cv2
import platform
from PyQt5.QtCore import QObject, pyqtSignal
from typing import List, Dict, Optional, Tuple


class DeviceManager(QObject):
    """设备管理类"""
    
    # 信号定义
    camera_detected = pyqtSignal(int, str)  # 相机检测信号（相机ID，设备名称）
    device_connected = pyqtSignal(str, bool)  # 设备连接信号（设备类型，连接状态）
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.available_cameras: List[Dict] = []
        
    def detect_cameras(self, max_cameras: int = 10) -> List[Dict]:
        """
        检测可用的相机设备
        
        Args:
            max_cameras: 最大检测数量
            
        Returns:
            List[Dict]: 可用相机列表，每个元素包含 {'id': int, 'name': str, 'info': str}
        """
        self.available_cameras = []
        
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # 尝试读取一帧以确认相机可用
                ret, _ = cap.read()
                if ret:
                    # 获取相机信息
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    info = f"分辨率: {width}x{height}"
                    
                    camera_info = {
                        'id': i,
                        'name': f"相机 {i}",
                        'info': info
                    }
                    self.available_cameras.append(camera_info)
                    self.camera_detected.emit(i, f"相机 {i}")
                cap.release()
        
        return self.available_cameras
    
    def get_camera_info(self, camera_id: int) -> Optional[Dict]:
        """
        获取指定相机的详细信息
        
        Args:
            camera_id: 相机ID
            
        Returns:
            Dict: 相机信息，如果相机不可用返回None
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            return None
        
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
            saturation = cap.get(cv2.CAP_PROP_SATURATION)
            
            info = {
                'id': camera_id,
                'width': width,
                'height': height,
                'fps': fps,
                'brightness': brightness,
                'saturation': saturation,
                'resolution': f"{width}x{height}",
                'status': '可用'
            }
            return info
        except Exception as e:
            print(f"[设备管理] 获取相机信息失败: {e}")
            return None
        finally:
            cap.release()
    
    def test_camera_connection(self, camera_id: int) -> Tuple[bool, str]:
        """
        测试相机连接
        
        Args:
            camera_id: 相机ID
            
        Returns:
            Tuple[bool, str]: (是否成功, 消息)
        """
        try:
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                return False, f"无法打开相机 {camera_id}"
            
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                return True, f"相机 {camera_id} 连接成功"
            else:
                return False, f"相机 {camera_id} 无法读取画面"
        except Exception as e:
            return False, f"测试相机连接时发生错误: {str(e)}"
    
    def test_alarm_light(self, light_type: str, address: str = "", port: int = 9600) -> Tuple[bool, str]:
        """
        测试报警灯连接
        
        Args:
            light_type: 报警灯类型（serial/usb/network）
            address: 连接地址
            port: 端口号
            
        Returns:
            Tuple[bool, str]: (是否成功, 消息)
        """
        # TODO: 实现报警灯连接测试
        print(f"[设备管理] 测试报警灯连接: 类型={light_type}, 地址={address}, 端口={port}")
        
        # 占位实现
        if light_type == "serial":
            if not address:
                return False, "请设置串口地址"
            # TODO: 实现串口连接测试
            return True, f"报警灯（串口 {address}）连接成功"
        elif light_type == "usb":
            if not address:
                return False, "请设置USB设备地址"
            # TODO: 实现USB连接测试
            return True, f"报警灯（USB {address}）连接成功"
        elif light_type == "network":
            if not address:
                return False, "请设置网络地址"
            # TODO: 实现网络连接测试
            return True, f"报警灯（网络 {address}:{port}）连接成功"
        else:
            return False, "不支持的报警灯类型"
    
    def get_available_ports(self) -> List[str]:
        """
        获取可用的串口/USB端口列表
        
        Returns:
            List[str]: 可用端口列表
        """
        # TODO: 实现端口检测（需要pyserial库）
        # 占位实现
        return ["COM1", "COM2", "COM3", "COM4"]

