"""
OCR处理模块
负责OCR识别、框选、文本检测、区域管理等功能
"""
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

from PyQt5.QtCore import QObject, pyqtSignal

# 将 method 目录加入路径以便导入 DetInfer / RecInfer
_method_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'method')
if _method_dir not in sys.path:
    sys.path.insert(0, _method_dir)


@dataclass
class TextRegion:
    """检测到的文本区域，用于后续识别监测"""
    id: int
    polygon: List[Tuple[float, float]]  # 4 个顶点坐标 (x,y)，图像坐标系
    from_mode: str  # "area" | "coord"
    score: float = 0.0
    meta: dict = field(default_factory=dict)


class OCRProcessor(QObject):
    """OCR处理类"""
    
    # 信号定义
    recognition_started = pyqtSignal()
    recognition_stopped = pyqtSignal()
    target_selected = pyqtSignal()
    area_selected = pyqtSignal()
    regions_updated = pyqtSignal(list)  # 检测完成，发出 TextRegion 列表
    regions_cleared = pyqtSignal()  # 刷新提示后发出
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_recognizing = False
        self.det_model = None
        self.rec_model = None
        self.text_regions: List[TextRegion] = []
        self._region_id_counter = 0
        
    def load_det_model(self, model_path: str) -> bool:
        """加载文本检测模型"""
        print(f"[OCR] 加载检测模型: {model_path}")
        if not model_path or not isinstance(model_path, str) or not os.path.isfile(model_path):
            print(f"[OCR] 检测模型路径无效: {model_path}")
            return False
        try:
            from det_infer import DetInfer
            self.det_model = DetInfer(model_path)
            print(f"[OCR] 检测模型已加载: {model_path}")
            return True
        except Exception as e:
            print(f"[OCR] 加载检测模型失败: {e}")
            self.det_model = None
            return False
    
    def has_det_model(self) -> bool:
        """是否已加载检测模型"""
        return self.det_model is not None
    
    def load_rec_model(self, model_path: str, dict_path: str) -> bool:
        """加载文本识别模型"""
        if not model_path or not isinstance(model_path, str) or not os.path.isfile(model_path):
            print(f"[OCR] 识别模型路径无效: {model_path}")
            return False
        if not dict_path or not isinstance(dict_path, str) or not os.path.isfile(dict_path):
            print(f"[OCR] 识别字典路径无效: {dict_path}")
            return False
        try:
            from rec_infer import RecInfer
            self.rec_model = RecInfer(model_path, dict_path)
            print(f"[OCR] 识别模型已加载: {model_path}")
            return True
        except Exception as e:
            print(f"[OCR] 加载识别模型失败: {e}")
            self.rec_model = None
            return False
    
    def has_rec_model(self) -> bool:
        """是否已加载识别模型"""
        return self.rec_model is not None
    
    def recognize_regions(self, frame, conf_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        对当前所有文本区域进行识别
        返回: [{"region": TextRegion, "text": str, "score": float, "numeric": float|None}, ...]
        - score >= conf_threshold 且可解析为数字时，才写入监控记录
        - 低置信度在 UI 显示 "/"，非纯数字区域不参与数值预警
        """
        if not self.has_rec_model():
            return []
        regions = self.get_regions()
        if not regions:
            return []
        try:
            from rec_infer import get_rotate_crop_image
            import numpy as np
        except ImportError as e:
            print(f"[OCR] 导入 rec_infer 失败: {e}")
            return []
        imgs = []
        for r in regions:
            pts = np.array(r.polygon, dtype=np.float32)
            try:
                crop = get_rotate_crop_image(frame, pts)
                imgs.append(crop)
            except Exception as e:
                print(f"[OCR] 裁剪区域 {r.id} 失败: {e}")
                imgs.append(np.zeros((32, 32, 3), dtype=np.uint8))
        if not imgs:
            return []
        try:
            out_txts = self.rec_model.predict(imgs)
        except Exception as e:
            print(f"[OCR] 识别异常: {e}")
            return []
        results = []
        for i, (region, raw) in enumerate(zip(regions, out_txts)):

            text, conf_list = raw[0]
            score = float(np.mean(conf_list)) if conf_list else 0.0
            numeric = self._extract_numeric(text)
            results.append({
                "region": region,
                "text": text,
                "score": score,
                "numeric": numeric
            })
        # print(f"[OCR] 识别结果: {results}")
        return results
    
    def _extract_numeric(self, text: str) -> Optional[float]:
        """从文本中提取第一个可解析的数字，失败返回 None"""
        if not text or not isinstance(text, str):
            return None
        m = re.search(r"[-+]?[0-9]*\.?[0-9]+", text)
        if m:
            try:
                return float(m.group())
            except ValueError:
                return None
        return None
    
    def detect_in_roi(self, frame, roi_rect: Tuple[int, int, int, int]) -> List[TextRegion]:
        """
        在矩形 ROI 内进行文本检测
        roi_rect: (x, y, w, h) 帧坐标系
        返回检测到的文本区域列表，并更新 text_regions、发出 regions_updated
        """
        if not self.has_det_model():
            print("[OCR] 检测模型未加载")
            return []
        x, y, w, h = roi_rect
        h_frame, w_frame = frame.shape[:2]
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = max(1, min(w, w_frame - x))
        h = max(1, min(h, h_frame - y))
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0:
            return []
        try:
            box_list, score_list = self.det_model.predict(crop)
        except Exception as e:
            print(f"[OCR] ROI 检测异常: {e}")
            return []
        regions: List[TextRegion] = []
        for i, (box, score) in enumerate(zip(box_list, score_list)):
            # 将 ROI 相对坐标平移回全图坐标
            poly = [(float(p[0]) + x, float(p[1]) + y) for p in box]
            self._region_id_counter += 1
            regions.append(TextRegion(
                id=self._region_id_counter,
                polygon=poly,
                from_mode="area",
                score=float(score)
            ))
        # 追加到已有区域，而不是清空
        self.text_regions.extend(regions)
        self.regions_updated.emit(self.get_regions())
        return regions
    
    def detect_with_points(self, frame, points: List[Tuple[float, float]]) -> List[TextRegion]:
        """
        全图检测，仅保留包含任意提示点的文本区域
        points: 帧坐标系下的提示点列表 [(x1,y1), (x2,y2), ...]
        """
        if not self.has_det_model():
            print("[OCR] 检测模型未加载")
            return []
        if not points:
            return []
        try:
            box_list, score_list = self.det_model.predict(frame)
        except Exception as e:
            print(f"[OCR] 全图检测异常: {e}")
            return []
        regions: List[TextRegion] = []
        for i, (box, score) in enumerate(zip(box_list, score_list)):
            if self._polygon_contains_any_point(box, points):
                poly = [(float(p[0]), float(p[1])) for p in box]
                self._region_id_counter += 1
                regions.append(TextRegion(
                    id=self._region_id_counter,
                    polygon=poly,
                    from_mode="coord",
                    score=float(score)
                ))
        # 追加到已有区域，而不是清空
        self.text_regions.extend(regions)
        self.regions_updated.emit(self.get_regions())
        return regions
    
    def _polygon_contains_any_point(self, polygon, points) -> bool:
        """判断多边形是否包含任意一个点（使用 cv2.pointPolygonTest）"""
        try:
            import cv2
            import numpy as np
            pts = np.array(polygon, dtype=np.float32)
            for px, py in points:
                if cv2.pointPolygonTest(pts, (px, py), False) >= 0:
                    return True
            return False
        except Exception:
            # 降级：用外接矩形判断
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            for px, py in points:
                if xmin <= px <= xmax and ymin <= py <= ymax:
                    return True
            return False
    
    def clear_regions(self):
        """清除所有文本区域，发出 regions_cleared"""
        self.text_regions = []
        self._region_id_counter = 0
        self.regions_cleared.emit()

    def add_manual_region(self, polygon: List[Tuple[float, float]], from_mode: str = "manual",
                          score: float = 1.0) -> TextRegion:
        """
        手动添加一个文本区域（例如主界面框选目标得到的四边形）
        polygon: 帧坐标系下的顶点列表
        """
        if not polygon:
            raise ValueError("polygon is empty")
        self._region_id_counter += 1
        region = TextRegion(
            id=self._region_id_counter,
            polygon=[(float(x), float(y)) for x, y in polygon],
            from_mode=from_mode,
            score=float(score),
        )
        self.text_regions.append(region)
        # 发出完整区域列表，便于主界面刷新覆盖层
        self.regions_updated.emit(self.get_regions())
        return region
    
    def get_regions(self) -> List[TextRegion]:
        """获取当前文本区域列表，供后续识别监测使用"""
        return list(self.text_regions)
        
    def select_target(self):
        """框选目标（占位实现）"""
        print("[OCR] 框选目标")
        self.target_selected.emit()
        
    def select_area(self):
        """框选区域（占位实现，实际由主窗口进入区域提示模式）"""
        print("[OCR] 框选区域")
        self.area_selected.emit()
        
    def click_hint(self):
        """点击提示/坐标提示（占位，实际由主窗口进入坐标提示模式）"""
        print("[OCR] 点击提示")
        
    def global_search(self):
        """全局搜索（占位实现）"""
        print("[OCR] 全局搜索")
        
    def start_recognition(self):
        """开始识别（占位实现）"""
        print("[OCR] 开始识别")
        self.is_recognizing = True
        self.recognition_started.emit()
        
    def stop_recognition(self):
        """结束识别（占位实现）"""
        print("[OCR] 结束识别")
        self.is_recognizing = False
        self.recognition_stopped.emit()
        
    def is_processing(self):
        """检查是否正在识别"""
        return self.is_recognizing
