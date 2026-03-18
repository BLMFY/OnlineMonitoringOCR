"""
日志管理模块
负责监控记录、报警记录的管理和预警阈值配置
"""
from PyQt5.QtCore import QObject, pyqtSignal, QDateTime
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class MonitoringRecord:
    """监控记录数据结构"""
    id: str
    timestamp: datetime
    area_name: str
    ocr_value: float
    ocr_text: str = ""
    status: str = "normal"  # normal/warning/alarm
    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None
    remark: str = ""  # 备注


@dataclass
class AlarmRecord:
    """报警记录数据结构"""
    id: str
    timestamp: datetime
    area_name: str
    ocr_value: float
    threshold_min: float
    threshold_max: float
    alarm_type: str  # above_max/below_min
    processed: bool = False
    processed_time: Optional[datetime] = None
    remark: str = ""  # 备注


@dataclass
class ThresholdConfig:
    """预警阈值配置数据结构"""
    id: str  # 配置ID（同时作为区域标识）
    min_value: float
    max_value: float
    enabled: bool = True
    remark: str = ""  # 备注
    created_time: datetime = field(default_factory=datetime.now)
    updated_time: datetime = field(default_factory=datetime.now)


class LogManager(QObject):
    """日志管理类"""
    
    # 信号定义
    record_added = pyqtSignal(MonitoringRecord)  # 新记录添加信号
    alarm_triggered = pyqtSignal(AlarmRecord)  # 报警触发信号
    threshold_changed = pyqtSignal(str)  # 阈值配置变更信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # 使用列表存储记录（后续可改为数据库）
        self.monitoring_records: List[MonitoringRecord] = []
        self.alarm_records: List[AlarmRecord] = []
        self.threshold_configs: Dict[str, ThresholdConfig] = {}  # key为配置ID
        
    def add_monitoring_record(self, area_name: str, ocr_value: float, 
                             ocr_text: str = "", timestamp: Optional[datetime] = None,
                             remark: str = ""):
        """
        添加监控记录
        
        Args:
            area_name: 区域名称
            ocr_value: OCR识别的数值
            ocr_text: OCR识别的完整文本
            timestamp: 时间戳，如果为None则使用当前时间
            remark: 备注
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # 检查是否触发报警
        status = "normal"
        threshold_min = None
        threshold_max = None
        
        # 查找该区域的配置
        config = self.get_threshold_config_by_area(area_name)
        if config and config.enabled:
            threshold_min = config.min_value
            threshold_max = config.max_value
            
            if ocr_value > config.max_value:
                status = "alarm"
                self._trigger_alarm(area_name, ocr_value, threshold_min, 
                                threshold_max, "above_max", timestamp, remark)
            elif ocr_value < config.min_value:
                status = "alarm"
                self._trigger_alarm(area_name, ocr_value, threshold_min, 
                                threshold_max, "below_min", timestamp, remark)
        elif config and (ocr_value > config.max_value * 0.9 or 
                ocr_value < config.min_value * 1.1):
            status = "warning"
        
        record = MonitoringRecord(
            id=f"MR_{timestamp.strftime('%Y%m%d%H%M%S%f')}",
            timestamp=timestamp,
            area_name=area_name,
            ocr_value=ocr_value,
            ocr_text=ocr_text,
            status=status,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            remark=remark
        )
        
        self.monitoring_records.append(record)
        self.record_added.emit(record)
        
        return record
    
    def _trigger_alarm(self, area_name: str, ocr_value: float, 
                      threshold_min: float, threshold_max: float,
                      alarm_type: str, timestamp: datetime, remark: str = ""):
        """触发报警"""
        alarm = AlarmRecord(
            id=f"AL_{timestamp.strftime('%Y%m%d%H%M%S%f')}",
            timestamp=timestamp,
            area_name=area_name,
            ocr_value=ocr_value,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            alarm_type=alarm_type,
            remark=remark
        )
        
        self.alarm_records.append(alarm)
        self.alarm_triggered.emit(alarm)
    
    def query_monitoring_records(self, start_time: Optional[datetime] = None,
                                end_time: Optional[datetime] = None,
                                area_name: Optional[str] = None,
                                status: Optional[str] = None) -> List[MonitoringRecord]:
        """
        查询监控记录
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            area_name: 区域名称（None表示所有区域）
            status: 状态筛选（normal/warning/alarm，None表示所有状态）
            
        Returns:
            List[MonitoringRecord]: 符合条件的记录列表
        """
        results = self.monitoring_records.copy()
        
        # 时间筛选
        if start_time:
            results = [r for r in results if r.timestamp >= start_time]
        if end_time:
            results = [r for r in results if r.timestamp <= end_time]
        
        # 区域筛选
        if area_name:
            results = [r for r in results if r.area_name == area_name]
        
        # 状态筛选
        if status:
            results = [r for r in results if r.status == status]
        
        # 按时间倒序排列
        results.sort(key=lambda x: x.timestamp, reverse=True)
        
        return results
    
    def query_alarm_records(self, start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None,
                           area_name: Optional[str] = None,
                           alarm_type: Optional[str] = None,
                           processed: Optional[bool] = None) -> List[AlarmRecord]:
        """
        查询报警记录
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            area_name: 区域名称
            alarm_type: 报警类型（above_max/below_min）
            processed: 是否已处理
            
        Returns:
            List[AlarmRecord]: 符合条件的报警记录列表
        """
        results = self.alarm_records.copy()
        
        # 时间筛选
        if start_time:
            results = [r for r in results if r.timestamp >= start_time]
        if end_time:
            results = [r for r in results if r.timestamp <= end_time]
        
        # 区域筛选
        if area_name:
            results = [r for r in results if r.area_name == area_name]
        
        # 报警类型筛选
        if alarm_type:
            results = [r for r in results if r.alarm_type == alarm_type]
        
        # 处理状态筛选
        if processed is not None:
            results = [r for r in results if r.processed == processed]
        
        # 按时间倒序排列
        results.sort(key=lambda x: x.timestamp, reverse=True)
        
        return results
    
    def get_recent_records(self, count: int = 100) -> List[MonitoringRecord]:
        """
        获取最近的监控记录
        
        Args:
            count: 记录数量
            
        Returns:
            List[MonitoringRecord]: 最近的记录列表
        """
        sorted_records = sorted(self.monitoring_records, 
                               key=lambda x: x.timestamp, reverse=True)
        return sorted_records[:count]
    
    def get_recent_alarms(self, count: int = 50) -> List[AlarmRecord]:
        """
        获取最近的报警记录
        
        Args:
            count: 记录数量
            
        Returns:
            List[AlarmRecord]: 最近的报警记录列表
        """
        sorted_alarms = sorted(self.alarm_records, 
                              key=lambda x: x.timestamp, reverse=True)
        return sorted_alarms[:count]
    
    def mark_alarm_processed(self, alarm_id: str):
        """
        标记报警为已处理
        
        Args:
            alarm_id: 报警记录ID
        """
        for alarm in self.alarm_records:
            if alarm.id == alarm_id:
                alarm.processed = True
                alarm.processed_time = datetime.now()
                break
    
    def add_threshold_config(self, config_id: str, min_value: float, 
                            max_value: float, enabled: bool = True, remark: str = ""):
        """
        添加或更新预警阈值配置
        
        Args:
            config_id: 配置ID（同时作为区域标识）
            min_value: 下限值
            max_value: 上限值
            enabled: 是否启用
            remark: 备注
        """
        if config_id in self.threshold_configs:
            # 更新现有配置
            config = self.threshold_configs[config_id]
            config.min_value = min_value
            config.max_value = max_value
            config.enabled = enabled
            config.remark = remark
            config.updated_time = datetime.now()
        else:
            # 创建新配置
            config = ThresholdConfig(
                id=config_id,
                min_value=min_value,
                max_value=max_value,
                enabled=enabled,
                remark=remark
            )
            self.threshold_configs[config_id] = config
        
        self.threshold_changed.emit(config_id)
    
    def remove_threshold_config(self, config_id: str):
        """
        删除预警阈值配置
        
        Args:
            config_id: 配置ID
        """
        if config_id in self.threshold_configs:
            del self.threshold_configs[config_id]
            self.threshold_changed.emit(config_id)
    
    def get_threshold_config(self, config_id: str) -> Optional[ThresholdConfig]:
        """
        获取预警阈值配置
        
        Args:
            config_id: 配置ID
            
        Returns:
            ThresholdConfig: 配置对象，如果不存在返回None
        """
        return self.threshold_configs.get(config_id)
    
    def get_threshold_config_by_area(self, area_name: str) -> Optional[ThresholdConfig]:
        """
        根据区域名称获取预警阈值配置
        注意：现在配置ID就是区域标识，所以直接通过ID查找
        
        Args:
            area_name: 区域名称（实际是配置ID）
            
        Returns:
            ThresholdConfig: 配置对象，如果不存在返回None
        """
        return self.threshold_configs.get(area_name)
    
    def get_all_threshold_configs(self) -> List[ThresholdConfig]:
        """
        获取所有预警阈值配置
        
        Returns:
            List[ThresholdConfig]: 所有配置列表
        """
        return list(self.threshold_configs.values())
    
    def clear_monitoring_records(self):
        """清空所有监控记录"""
        self.monitoring_records.clear()
    
    def clear_alarm_records(self):
        """清空所有报警记录"""
        self.alarm_records.clear()
    
    def clear_threshold_configs(self):
        """清空所有预警阈值配置（主界面刷新提示时调用）"""
        self.threshold_configs.clear()

