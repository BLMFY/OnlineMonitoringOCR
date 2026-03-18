"""
模型管理模块
负责模型列表管理和模型信息获取
"""
import os
from PyQt5.QtCore import QObject, pyqtSignal
from typing import Dict, List, Optional


def _project_root():
    """项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ModelManager(QObject):
    """模型管理类"""
    
    # 信号定义
    model_changed = pyqtSignal(str, str)  # 模型变更信号（模型类型，模型名称）
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 预配置的文字检测模型
        self.text_detection_models = {
            "model_1": {
                "name": "PTDet检测模型",
                "description": "自适应文本检测",
                "confidence_default": 0.5
            },
            "model_2": {
                "name": "DBNet检测模型",
                "description": "平衡检测模型，速度和精度兼顾",
                "confidence_default": 0.6
            },
            "model_3": {
                "name": "文字检测模型3",
                "description": "高精度检测模型，适合复杂场景",
                "confidence_default": 0.7
            }
        }
        
        # 预配置的OCR识别模型
        self.ocr_models = {
            "ocr_model_1": {
                "name": "通用OCR识别模型",
                "description": "通用OCR模型，支持中英文",
                "languages": ["中文", "英文"]
            },
            "ocr_model_2": {
                "name": "晶体数码管模型",
                "description": "高精度OCR模型，支持多语言",
                "languages": ["英文", "数字"]
            },
            "ocr_model_3": {
                "name": "轻量OCR识别模型",
                "description": "快速OCR模型，适合实时识别",
                "languages": ["中文", "英文"]
            }
        }
        
        # 预配置的图像增强模型
        self.enhancement_models = {
            "enhance_model_1": {
                "name": "ECCE增强模型",
                "description": "基础增强模型",
                "strength_options": ["medium", "strong"]
            },
            "enhance_model_2": {
                "name": "Zero-DCE增强模型",
                "description": "高级增强模型",
                "strength_options": ["medium", "strong"]
            }
        }
        
        # 文本检测模型权重路径（相对于 method/weight/det/）
        self.text_detection_model_paths = {
            "model_1": "det_db_mbv3_new.pth",  # PTDet检测模型（默认）
            "model_2": "det_db.pth",
            "model_3": "det_model3.pth",
        }
        
        # OCR识别模型权重与字典路径（相对于 method/weight/rec/）
        self.ocr_model_paths = {
            "ocr_model_1": {"weight": "chen/chen_crnn_mbv3.pth", "dict": "chen/chen.txt"},  # 通用OCR识别模型（默认）
            "ocr_model_2": {"weight": "numtube/numtube_rec.pth", "dict": "numtube/digital_dict.txt"},
            "ocr_model_3": {"weight": "xxx/xxx.pth", "dict": "xxx/xxx.txt"},
        }
    
    def get_text_detection_models(self) -> Dict:
        """获取文字检测模型列表"""
        return self.text_detection_models
    
    def get_ocr_models(self) -> Dict:
        """获取OCR识别模型列表"""
        return self.ocr_models
    
    def get_enhancement_models(self) -> Dict:
        """获取图像增强模型列表"""
        return self.enhancement_models
    
    def get_model_info(self, model_type: str, model_id: str) -> Optional[Dict]:
        """
        获取模型信息
        
        Args:
            model_type: 模型类型（text_detection/ocr/enhancement）
            model_id: 模型ID
            
        Returns:
            Dict: 模型信息，如果不存在返回None
        """
        if model_type == "text_detection":
            return self.text_detection_models.get(model_id)
        elif model_type == "ocr":
            return self.ocr_models.get(model_id)
        elif model_type == "enhancement":
            return self.enhancement_models.get(model_id)
        return None
    
    def get_det_model_path(self, model_id: str) -> Optional[str]:
        """
        获取文本检测模型权重文件的绝对路径
        model_id: 如 model_1, model_2
        返回: 绝对路径，若未配置则返回 None
        """
        rel = self.text_detection_model_paths.get(model_id)
        if not rel:
            return None
        return os.path.join(_project_root(), "method", "weight", "det", rel)
    
    def get_ocr_model_path(self, model_id: str) -> Optional[str]:
        """
        获取OCR识别模型权重文件的绝对路径
        model_id: 如 ocr_model_1
        返回: 绝对路径，若未配置则返回 None
        """
        info = self.ocr_model_paths.get(model_id)
        if not info or "weight" not in info:
            return None
        return os.path.join(_project_root(), "method", "weight", "rec", info["weight"])
    
    def get_ocr_dict_path(self, model_id: str) -> Optional[str]:
        """
        获取OCR识别模型字典文件的绝对路径
        model_id: 如 ocr_model_1
        返回: 绝对路径，若未配置则返回 None
        """
        info = self.ocr_model_paths.get(model_id)
        if not info or "dict" not in info:
            return None
        return os.path.join(_project_root(), "method", "weight", "rec", info["dict"])
    
    def validate_model(self, model_type: str, model_id: str) -> bool:
        """
        验证模型是否存在
        
        Args:
            model_type: 模型类型
            model_id: 模型ID
            
        Returns:
            bool: 模型是否存在
        """
        return self.get_model_info(model_type, model_id) is not None

