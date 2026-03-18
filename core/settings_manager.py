"""
设置管理模块
负责系统配置的加载、保存和管理
"""
import json
import os
from PyQt5.QtCore import QObject, pyqtSignal
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict


@dataclass
class SystemConfig:
    """系统配置数据结构"""
    # 相机配置
    camera_id: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    # 测试模式（启用时主界面显示测试素材而非真实相机）
    test_mode_enabled: bool = False
    test_material_path: str = ""  # 图像或视频文件路径
    camera_brightness: float = 0.5
    camera_saturation: float = 0.5
    
    # 硬件配置
    alarm_light_type: str = "serial"  # serial/usb/network
    alarm_light_address: str = ""
    alarm_light_port: int = 9600
    alarm_light_mode: str = "flash"  # always/flash/sound
    alarm_light_flash_frequency: int = 2  # 闪烁频率（次/秒）
    
    # 模型配置
    text_detection_model: str = "model_1"  # 预配置的模型名称
    text_detection_confidence: float = 0.5
    
    ocr_model: str = "ocr_model_1"  # 预配置的OCR模型名称
    
    enhancement_model: str = "enhance_model_1"  # 预配置的增强模型名称
    enhancement_strength: str = "medium"  # medium/strong
    
    # 存储路径
    log_path: str = "./logs"
    data_path: str = "./data"


class SettingsManager(QObject):
    """设置管理类"""
    
    # 信号定义
    config_changed = pyqtSignal()  # 配置变更信号
    config_saved = pyqtSignal()  # 配置保存信号
    
    def __init__(self, config_file: str = "config.json", parent=None):
        super().__init__(parent)
        self.config_file = config_file
        self.config = SystemConfig()
        self.load_config()
        
        # 预配置的模型列表
        self.text_detection_models = {
            "model_1": "PTDet检测模型",
            "model_2": "DBNet检测模型"
        }
        
        self.ocr_models = {
            "ocr_model_1": "通用OCR识别模型",
            "ocr_model_2": "晶体数码管模型",
            "ocr_model_3": "轻量OCR识别模型"
        }
        
        self.enhancement_models = {
            "enhance_model_1": "ECCE增强模型",
            "enhance_model_2": "Zero-DCE增强模型"
        }
    
    def load_config(self):
        """从文件加载配置，并对关键路径做保护"""
        # 确保 config_file 是字符串类型
        if not isinstance(self.config_file, str):
            self.config_file = "config.json"
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 更新配置对象
                    for key, value in data.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
                print(f"[设置管理] 配置已从 {self.config_file} 加载")
            except Exception as e:
                print(f"[设置管理] 加载配置失败: {e}")
                self.config = SystemConfig()  # 使用默认配置
        else:
            print("[设置管理] 配置文件不存在，使用默认配置")
            self.config = SystemConfig()
        
        # 对关键路径做保护，防止为空导致程序异常
        if not self.config.log_path or not isinstance(self.config.log_path, str):
            self.config.log_path = "./logs"
        if not self.config.data_path or not isinstance(self.config.data_path, str):
            self.config.data_path = "./data"
    
    def save_config(self):
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.config), f, indent=4, ensure_ascii=False)
            print(f"[设置管理] 配置已保存到 {self.config_file}")
            self.config_saved.emit()
            return True
        except Exception as e:
            print(f"[设置管理] 保存配置失败: {e}")
            return False
    
    def get_config(self) -> SystemConfig:
        """获取当前配置"""
        return self.config
    
    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.config_changed.emit()
    
    def reset_to_default(self):
        """重置为默认配置"""
        self.config = SystemConfig()
        self.config_changed.emit()
        print("[设置管理] 配置已重置为默认值")
    
    def get_text_detection_models(self) -> Dict[str, str]:
        """获取文字检测模型列表"""
        return self.text_detection_models
    
    def get_ocr_models(self) -> Dict[str, str]:
        """获取OCR识别模型列表"""
        return self.ocr_models
    
    def get_enhancement_models(self) -> Dict[str, str]:
        """获取图像增强模型列表"""
        return self.enhancement_models

