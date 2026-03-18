"""
日志界面窗口
包含监控记录、报警记录查看和预警阈值设置
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
                              QTableWidget, QTableWidgetItem, QGroupBox, QPushButton,
                              QLabel, QDateTimeEdit, QComboBox, QLineEdit, QCheckBox,
                              QMessageBox, QFileDialog, QHeaderView, QAbstractItemView)
from PyQt5.QtCore import Qt, QDateTime, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QBrush
from datetime import datetime, timedelta
from typing import List, Optional

from core.log_manager import LogManager, MonitoringRecord, AlarmRecord, ThresholdConfig


class LogWindow(QWidget):
    """日志界面窗口类"""
    
    def __init__(self, log_manager: LogManager, parent=None):
        super().__init__(parent)
        self.log_manager = log_manager
        self._alarm_popup_cooldown: dict = {}  # (area_name, alarm_type) -> last_shown_time
        
        self.init_ui()
        self.connect_signals()
        self.load_recent_records()
        self.load_threshold_configs()
    
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 创建标签页（顺序：监测动态、预警设置、监控记录、报警记录）
        self.tab_widget = QTabWidget()
        self.tab_widget.setFont(QFont("Microsoft YaHei", 10))
        
        # 监测动态标签页（第一位，实时识别结果）
        self.live_tab = self.create_live_tab()
        self.tab_widget.insertTab(0, self.live_tab, "监测动态")
        
        # 预警设置标签页（第二位）
        self.threshold_tab = self.create_threshold_tab()
        self.tab_widget.insertTab(1, self.threshold_tab, "预警设置")
        
        # 监控记录标签页
        self.monitoring_tab = self.create_monitoring_tab()
        self.tab_widget.addTab(self.monitoring_tab, "监控记录")
        
        # 报警记录标签页
        self.alarm_tab = self.create_alarm_tab()
        self.tab_widget.addTab(self.alarm_tab, "报警记录")
        
        layout.addWidget(self.tab_widget)
    
    def create_live_tab(self):
        """创建监测动态标签页（实时识别结果）"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        tip = QLabel("实时显示当前监测目标的识别结果，监测开启时自动更新")
        tip.setFont(QFont("Microsoft YaHei", 9))
        tip.setStyleSheet("color: #7f8c8d;")
        layout.addWidget(tip)
        
        self.live_table = QTableWidget()
        self.live_table.setColumnCount(5)
        self.live_table.setHorizontalHeaderLabels(
            ["目标序号", "识别值", "识别文本", "状态", "备注"]
        )
        self.live_table.setFont(QFont("Microsoft YaHei", 9))
        self.live_table.setAlternatingRowColors(True)
        self.live_table.horizontalHeader().setStretchLastSection(True)
        self.live_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.live_table)
        
        return widget
    
    def update_live_display(self, rows: List[tuple]):
        """
        更新监测动态表格
        rows: [(目标序号, 识别值, 识别文本, 状态, 备注), ...]
        状态: 正常 / 警告：超出下限 / 警告：超出上限
        """
        self.live_table.setRowCount(len(rows))
        for row_idx, (target_id, value, text, status, remark) in enumerate(rows):
            self.live_table.setItem(row_idx, 0, QTableWidgetItem(str(target_id)))
            self.live_table.setItem(row_idx, 1, QTableWidgetItem(str(value)))
            self.live_table.setItem(row_idx, 2, QTableWidgetItem(str(text)))
            status_item = QTableWidgetItem(status)
            if status == "警告：超出上限":
                status_item.setBackground(QBrush(QColor("#fff3cd")))
            elif status == "警告：超出下限":
                status_item.setBackground(QBrush(QColor("#fff3cd")))
            else:
                status_item.setBackground(QBrush(QColor("#d4edda")))
            self.live_table.setItem(row_idx, 3, status_item)
            self.live_table.setItem(row_idx, 4, QTableWidgetItem(str(remark)))
    
    def create_monitoring_tab(self):
        """创建监控记录标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 查询条件区域
        query_group = QGroupBox("查询条件")
        query_group.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        query_layout = QHBoxLayout(query_group)
        query_layout.setSpacing(10)
        
        # 开始时间
        query_layout.addWidget(QLabel("开始时间:"))
        self.monitoring_start_time = QDateTimeEdit()
        self.monitoring_start_time.setCalendarPopup(True)
        self.monitoring_start_time.setDateTime(QDateTime.currentDateTime().addDays(-7))
        self.monitoring_start_time.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.monitoring_start_time.setFont(QFont("Microsoft YaHei", 9))
        query_layout.addWidget(self.monitoring_start_time)
        
        # 结束时间
        query_layout.addWidget(QLabel("结束时间:"))
        self.monitoring_end_time = QDateTimeEdit()
        self.monitoring_end_time.setCalendarPopup(True)
        self.monitoring_end_time.setDateTime(QDateTime.currentDateTime())
        self.monitoring_end_time.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.monitoring_end_time.setFont(QFont("Microsoft YaHei", 9))
        query_layout.addWidget(self.monitoring_end_time)
        
        # 区域选择
        query_layout.addWidget(QLabel("区域:"))
        self.monitoring_area_combo = QComboBox()
        self.monitoring_area_combo.setEditable(True)
        self.monitoring_area_combo.addItem("全部区域")
        self.monitoring_area_combo.setFont(QFont("Microsoft YaHei", 9))
        self.monitoring_area_combo.setMinimumWidth(120)
        query_layout.addWidget(self.monitoring_area_combo)
        
        # 状态筛选
        query_layout.addWidget(QLabel("状态:"))
        self.monitoring_status_combo = QComboBox()
        self.monitoring_status_combo.addItems(["全部", "正常", "预警", "报警"])
        self.monitoring_status_combo.setFont(QFont("Microsoft YaHei", 9))
        query_layout.addWidget(self.monitoring_status_combo)
        
        # 查询按钮
        btn_query = QPushButton("查询")
        btn_query.setFont(QFont("Microsoft YaHei", 9))
        btn_query.setMinimumWidth(80)
        btn_query.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 15px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        btn_query.clicked.connect(self.on_query_monitoring)
        query_layout.addWidget(btn_query)
        
        # 重置按钮
        btn_reset = QPushButton("重置")
        btn_reset.setFont(QFont("Microsoft YaHei", 9))
        btn_reset.setMinimumWidth(80)
        btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 15px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
        btn_reset.clicked.connect(self.on_reset_monitoring_query)
        query_layout.addWidget(btn_reset)
        
        query_layout.addStretch()
        layout.addWidget(query_group)
        
        # 记录列表
        self.monitoring_table = QTableWidget()
        self.monitoring_table.setColumnCount(5)
        # ID 列显示目标序号（区域/配置 ID）
        self.monitoring_table.setHorizontalHeaderLabels(
            ["目标序号", "识别值", "识别文本", "状态", "备注"]
        )
        self.monitoring_table.setFont(QFont("Microsoft YaHei", 9))
        self.monitoring_table.setAlternatingRowColors(True)
        self.monitoring_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.monitoring_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.monitoring_table.horizontalHeader().setStretchLastSection(True)
        self.monitoring_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.monitoring_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.monitoring_table)
        
        # 操作按钮区域
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        
        btn_export_selected = QPushButton("导出选中")
        btn_export_selected.setFont(QFont("Microsoft YaHei", 9))
        btn_export_selected.clicked.connect(self.on_export_monitoring_selected)
        btn_layout.addWidget(btn_export_selected)
        
        btn_export_all = QPushButton("导出全部")
        btn_export_all.setFont(QFont("Microsoft YaHei", 9))
        btn_export_all.clicked.connect(self.on_export_monitoring_all)
        btn_layout.addWidget(btn_export_all)
        
        btn_refresh = QPushButton("刷新")
        btn_refresh.setFont(QFont("Microsoft YaHei", 9))
        btn_refresh.clicked.connect(self.load_recent_records)
        btn_layout.addWidget(btn_refresh)
        
        btn_clear = QPushButton("清空")
        btn_clear.setFont(QFont("Microsoft YaHei", 9))
        btn_clear.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 15px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        btn_clear.clicked.connect(self.on_clear_monitoring_records)
        btn_layout.addWidget(btn_clear)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        return widget
    
    def create_alarm_tab(self):
        """创建报警记录标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 查询条件区域
        query_group = QGroupBox("查询条件")
        query_group.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        query_layout = QHBoxLayout(query_group)
        query_layout.setSpacing(10)
        
        # 开始时间
        query_layout.addWidget(QLabel("开始时间:"))
        self.alarm_start_time = QDateTimeEdit()
        self.alarm_start_time.setCalendarPopup(True)
        self.alarm_start_time.setDateTime(QDateTime.currentDateTime().addDays(-7))
        self.alarm_start_time.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.alarm_start_time.setFont(QFont("Microsoft YaHei", 9))
        query_layout.addWidget(self.alarm_start_time)
        
        # 结束时间
        query_layout.addWidget(QLabel("结束时间:"))
        self.alarm_end_time = QDateTimeEdit()
        self.alarm_end_time.setCalendarPopup(True)
        self.alarm_end_time.setDateTime(QDateTime.currentDateTime())
        self.alarm_end_time.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.alarm_end_time.setFont(QFont("Microsoft YaHei", 9))
        query_layout.addWidget(self.alarm_end_time)
        
        # 区域选择
        query_layout.addWidget(QLabel("区域:"))
        self.alarm_area_combo = QComboBox()
        self.alarm_area_combo.setEditable(True)
        self.alarm_area_combo.addItem("全部区域")
        self.alarm_area_combo.setFont(QFont("Microsoft YaHei", 9))
        self.alarm_area_combo.setMinimumWidth(120)
        query_layout.addWidget(self.alarm_area_combo)
        
        # 报警级别
        query_layout.addWidget(QLabel("报警级别:"))
        self.alarm_type_combo = QComboBox()
        self.alarm_type_combo.addItems(["全部", "超出上限", "低于下限"])
        self.alarm_type_combo.setFont(QFont("Microsoft YaHei", 9))
        query_layout.addWidget(self.alarm_type_combo)
        
        # 查询按钮
        btn_query = QPushButton("查询")
        btn_query.setFont(QFont("Microsoft YaHei", 9))
        btn_query.setMinimumWidth(80)
        btn_query.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 15px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        btn_query.clicked.connect(self.on_query_alarm)
        query_layout.addWidget(btn_query)
        
        # 重置按钮
        btn_reset = QPushButton("重置")
        btn_reset.setFont(QFont("Microsoft YaHei", 9))
        btn_reset.setMinimumWidth(80)
        btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 15px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
        btn_reset.clicked.connect(self.on_reset_alarm_query)
        query_layout.addWidget(btn_reset)
        
        query_layout.addStretch()
        layout.addWidget(query_group)
        
        # 报警记录列表
        self.alarm_table = QTableWidget()
        self.alarm_table.setColumnCount(7)
        self.alarm_table.setHorizontalHeaderLabels(
            ["ID", "报警时间", "识别值", "阈值上限", "阈值下限", "报警状态", "备注"]
        )
        self.alarm_table.setFont(QFont("Microsoft YaHei", 9))
        self.alarm_table.setAlternatingRowColors(True)
        self.alarm_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.alarm_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.alarm_table.horizontalHeader().setStretchLastSection(True)
        self.alarm_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.alarm_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.alarm_table)
        
        # 操作按钮区域
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        
        btn_export_selected = QPushButton("导出选中")
        btn_export_selected.setFont(QFont("Microsoft YaHei", 9))
        btn_export_selected.clicked.connect(self.on_export_alarm_selected)
        btn_layout.addWidget(btn_export_selected)
        
        btn_export_all = QPushButton("导出全部")
        btn_export_all.setFont(QFont("Microsoft YaHei", 9))
        btn_export_all.clicked.connect(self.on_export_alarm_all)
        btn_layout.addWidget(btn_export_all)
        
        btn_refresh = QPushButton("刷新")
        btn_refresh.setFont(QFont("Microsoft YaHei", 9))
        btn_refresh.clicked.connect(self.load_recent_alarms)
        btn_layout.addWidget(btn_refresh)
        
        btn_clear = QPushButton("清空")
        btn_clear.setFont(QFont("Microsoft YaHei", 9))
        btn_clear.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 15px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        btn_clear.clicked.connect(self.on_clear_alarm_records)
        btn_layout.addWidget(btn_clear)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        return widget
    
    def create_threshold_tab(self):
        """创建预警设置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 阈值设置区域
        setting_group = QGroupBox("区域预警阈值设置")
        setting_group.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        setting_layout = QVBoxLayout(setting_group)
        setting_layout.setSpacing(10)
        
        # 配置ID
        id_layout = QHBoxLayout()
        id_layout.addWidget(QLabel("目标序号:"))
        self.threshold_id_input = QLineEdit()
        self.threshold_id_input.setPlaceholderText("请输入目标序号（唯一标识）")
        self.threshold_id_input.setFont(QFont("Microsoft YaHei", 9))
        id_layout.addWidget(self.threshold_id_input)
        setting_layout.addLayout(id_layout)
        
        # 上限值
        max_layout = QHBoxLayout()
        max_layout.addWidget(QLabel("上限值:"))
        self.threshold_max_input = QLineEdit()
        self.threshold_max_input.setPlaceholderText("请输入上限值")
        self.threshold_max_input.setFont(QFont("Microsoft YaHei", 9))
        max_layout.addWidget(self.threshold_max_input)
        setting_layout.addLayout(max_layout)
        
        # 下限值
        min_layout = QHBoxLayout()
        min_layout.addWidget(QLabel("下限值:"))
        self.threshold_min_input = QLineEdit()
        self.threshold_min_input.setPlaceholderText("请输入下限值")
        self.threshold_min_input.setFont(QFont("Microsoft YaHei", 9))
        min_layout.addWidget(self.threshold_min_input)
        setting_layout.addLayout(min_layout)
        
        # 备注
        remark_layout = QHBoxLayout()
        remark_layout.addWidget(QLabel("备注:"))
        self.threshold_remark_input = QLineEdit()
        self.threshold_remark_input.setPlaceholderText("请输入备注信息（可选）")
        self.threshold_remark_input.setFont(QFont("Microsoft YaHei", 9))
        remark_layout.addWidget(self.threshold_remark_input)
        setting_layout.addLayout(remark_layout)
        
        # 启用状态
        enable_layout = QHBoxLayout()
        self.threshold_enable_check = QCheckBox("启用预警")
        self.threshold_enable_check.setChecked(True)
        self.threshold_enable_check.setFont(QFont("Microsoft YaHei", 9))
        enable_layout.addWidget(self.threshold_enable_check)
        enable_layout.addStretch()
        setting_layout.addLayout(enable_layout)
        
        # 操作按钮
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        
        btn_add = QPushButton("添加")
        btn_add.setFont(QFont("Microsoft YaHei", 9))
        btn_add.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 15px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        btn_add.clicked.connect(self.on_add_threshold)
        btn_layout.addWidget(btn_add)
        
        btn_update = QPushButton("修改")
        btn_update.setFont(QFont("Microsoft YaHei", 9))
        btn_update.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 15px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        btn_update.clicked.connect(self.on_update_threshold)
        btn_layout.addWidget(btn_update)
        
        btn_delete = QPushButton("删除")
        btn_delete.setFont(QFont("Microsoft YaHei", 9))
        btn_delete.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 15px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        btn_delete.clicked.connect(self.on_delete_threshold)
        btn_layout.addWidget(btn_delete)
        
        btn_clear = QPushButton("清空")
        btn_clear.setFont(QFont("Microsoft YaHei", 9))
        btn_clear.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 15px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
        btn_clear.clicked.connect(self.on_clear_threshold_input)
        btn_layout.addWidget(btn_clear)
        
        btn_layout.addStretch()
        setting_layout.addLayout(btn_layout)
        
        layout.addWidget(setting_group)
        
        # 已设置区域列表
        list_group = QGroupBox("已设置区域列表")
        list_group.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        list_layout = QVBoxLayout(list_group)
        
        self.threshold_table = QTableWidget()
        self.threshold_table.setColumnCount(4)
        self.threshold_table.setHorizontalHeaderLabels(
            ["目标序号", "阈值上限", "阈值下限", "备注"]
        )
        self.threshold_table.setFont(QFont("Microsoft YaHei", 9))
        self.threshold_table.setAlternatingRowColors(True)
        self.threshold_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.threshold_table.horizontalHeader().setStretchLastSection(True)
        self.threshold_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.threshold_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.threshold_table.itemDoubleClicked.connect(self.on_threshold_table_double_clicked)
        list_layout.addWidget(self.threshold_table)
        
        layout.addWidget(list_group)
        
        return widget
    
    def connect_signals(self):
        """连接信号和槽"""
        self.log_manager.record_added.connect(self.on_record_added)
        self.log_manager.alarm_triggered.connect(self.on_alarm_triggered)
        self.log_manager.threshold_changed.connect(self.on_threshold_changed)
    
    def load_recent_records(self):
        """加载最近的监控记录"""
        records = self.log_manager.get_recent_records(100)
        self.update_monitoring_table(records)
        self.update_area_combos()
    
    def load_recent_alarms(self):
        """加载最近的报警记录"""
        alarms = self.log_manager.get_recent_alarms(50)
        self.update_alarm_table(alarms)
        self.update_area_combos()
    
    def load_threshold_configs(self):
        """加载预警阈值配置"""
        configs = self.log_manager.get_all_threshold_configs()
        self.update_threshold_table(configs)
        self.update_area_combos()
    
    def update_monitoring_table(self, records: List[MonitoringRecord]):
        """更新监控记录表格"""
        self.monitoring_table.setRowCount(len(records))
        
        for row, record in enumerate(records):
            # 目标序号（区域/配置 ID），使用 area_name 与阈值配置保持一致
            self.monitoring_table.setItem(row, 0, QTableWidgetItem(str(record.area_name)))
            
            # 识别值
            value_str = f"{record.ocr_value:.2f}" if isinstance(record.ocr_value, float) else str(record.ocr_value)
            self.monitoring_table.setItem(row, 1, QTableWidgetItem(value_str))
            
            # 识别文本
            self.monitoring_table.setItem(row, 2, QTableWidgetItem(record.ocr_text))
            
            # 状态
            status_item = QTableWidgetItem()
            if record.status == "normal":
                status_item.setText("正常")
                status_item.setBackground(QBrush(QColor("#d4edda")))
            elif record.status == "warning":
                status_item.setText("预警")
                status_item.setBackground(QBrush(QColor("#fff3cd")))
            else:  # alarm
                status_item.setText("报警")
                status_item.setBackground(QBrush(QColor("#f8d7da")))
            self.monitoring_table.setItem(row, 3, status_item)
            
            # 备注
            self.monitoring_table.setItem(row, 4, QTableWidgetItem(record.remark))
    
    def update_alarm_table(self, alarms: List[AlarmRecord]):
        """更新报警记录表格"""
        self.alarm_table.setRowCount(len(alarms))
        
        for row, alarm in enumerate(alarms):
            # ID
            self.alarm_table.setItem(row, 0, QTableWidgetItem(alarm.id))
            
            # 报警时间
            time_str = alarm.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            self.alarm_table.setItem(row, 1, QTableWidgetItem(time_str))
            
            # 识别值
            value_str = f"{alarm.ocr_value:.2f}" if isinstance(alarm.ocr_value, float) else str(alarm.ocr_value)
            self.alarm_table.setItem(row, 2, QTableWidgetItem(value_str))
            
            # 阈值上限
            self.alarm_table.setItem(row, 3, QTableWidgetItem(f"{alarm.threshold_max:.2f}"))
            
            # 阈值下限
            self.alarm_table.setItem(row, 4, QTableWidgetItem(f"{alarm.threshold_min:.2f}"))
            
            # 报警状态
            status_str = "超出上限" if alarm.alarm_type == "above_max" else "低于下限"
            status_item = QTableWidgetItem(status_str)
            status_item.setBackground(QBrush(QColor("#f8d7da")))
            self.alarm_table.setItem(row, 5, status_item)
            
            # 备注
            self.alarm_table.setItem(row, 6, QTableWidgetItem(alarm.remark))
    
    def update_threshold_table(self, configs: List[ThresholdConfig]):
        """更新预警阈值配置表格"""
        self.threshold_table.setRowCount(len(configs))
        
        for row, config in enumerate(configs):
            # ID
            self.threshold_table.setItem(row, 0, QTableWidgetItem(config.id))
            
            # 阈值上限
            self.threshold_table.setItem(row, 1, QTableWidgetItem(f"{config.max_value:.2f}"))
            
            # 阈值下限
            self.threshold_table.setItem(row, 2, QTableWidgetItem(f"{config.min_value:.2f}"))
            
            # 备注
            self.threshold_table.setItem(row, 3, QTableWidgetItem(config.remark))
    
    def update_area_combos(self):
        """更新区域下拉框"""
        configs = self.log_manager.get_all_threshold_configs()
        # 配置ID就是区域标识
        areas = set([config.id for config in configs])
        
        # 更新监控记录区域下拉框
        current_text = self.monitoring_area_combo.currentText()
        self.monitoring_area_combo.clear()
        self.monitoring_area_combo.addItem("全部区域")
        for area in sorted(areas):
            self.monitoring_area_combo.addItem(area)
        if current_text and current_text != "全部区域":
            index = self.monitoring_area_combo.findText(current_text)
            if index >= 0:
                self.monitoring_area_combo.setCurrentIndex(index)
        
        # 更新报警记录区域下拉框
        current_text = self.alarm_area_combo.currentText()
        self.alarm_area_combo.clear()
        self.alarm_area_combo.addItem("全部区域")
        for area in sorted(areas):
            self.alarm_area_combo.addItem(area)
        if current_text and current_text != "全部区域":
            index = self.alarm_area_combo.findText(current_text)
            if index >= 0:
                self.alarm_area_combo.setCurrentIndex(index)
    
    # ========== 事件处理方法 ==========
    
    def on_query_monitoring(self):
        """查询监控记录"""
        start_time = self.monitoring_start_time.dateTime().toPyDateTime()
        end_time = self.monitoring_end_time.dateTime().toPyDateTime()
        
        area_name = None
        if self.monitoring_area_combo.currentText() != "全部区域":
            area_name = self.monitoring_area_combo.currentText()
        
        status = None
        status_text = self.monitoring_status_combo.currentText()
        if status_text == "正常":
            status = "normal"
        elif status_text == "预警":
            status = "warning"
        elif status_text == "报警":
            status = "alarm"
        
        records = self.log_manager.query_monitoring_records(
            start_time=start_time,
            end_time=end_time,
            area_name=area_name,
            status=status
        )
        
        self.update_monitoring_table(records)
    
    def on_reset_monitoring_query(self):
        """重置监控记录查询条件"""
        self.monitoring_start_time.setDateTime(QDateTime.currentDateTime().addDays(-7))
        self.monitoring_end_time.setDateTime(QDateTime.currentDateTime())
        self.monitoring_area_combo.setCurrentIndex(0)
        self.monitoring_status_combo.setCurrentIndex(0)
        self.load_recent_records()
    
    def on_query_alarm(self):
        """查询报警记录"""
        start_time = self.alarm_start_time.dateTime().toPyDateTime()
        end_time = self.alarm_end_time.dateTime().toPyDateTime()
        
        area_name = None
        if self.alarm_area_combo.currentText() != "全部区域":
            area_name = self.alarm_area_combo.currentText()
        
        alarm_type = None
        type_text = self.alarm_type_combo.currentText()
        if type_text == "超出上限":
            alarm_type = "above_max"
        elif type_text == "低于下限":
            alarm_type = "below_min"
        
        alarms = self.log_manager.query_alarm_records(
            start_time=start_time,
            end_time=end_time,
            area_name=area_name,
            alarm_type=alarm_type
        )
        
        self.update_alarm_table(alarms)
    
    def on_reset_alarm_query(self):
        """重置报警记录查询条件"""
        self.alarm_start_time.setDateTime(QDateTime.currentDateTime().addDays(-7))
        self.alarm_end_time.setDateTime(QDateTime.currentDateTime())
        self.alarm_area_combo.setCurrentIndex(0)
        self.alarm_type_combo.setCurrentIndex(0)
        self.load_recent_alarms()
    
    def _get_valid_target_ids(self) -> List[str]:
        """获取当前有效的目标序号列表（来自主界面的监测区域）"""
        parent = self.parent()
        if parent and hasattr(parent, "ocr_processor"):
            regions = parent.ocr_processor.get_regions()
            return [str(r.id) for r in regions]
        return []
    
    def on_add_threshold(self):
        """添加预警阈值"""
        config_id = self.threshold_id_input.text().strip()
        if not config_id:
            QMessageBox.warning(self, "警告", "请输入目标序号")
            return
        
        valid_ids = self._get_valid_target_ids()
        if valid_ids and config_id not in valid_ids:
            ids_str = "、".join(valid_ids)
            QMessageBox.warning(
                self, "警告",
                f"目标序号 {config_id} 超出当前范围。\n当前有效序号为：{ids_str}"
            )
            return
        
        try:
            max_value = float(self.threshold_max_input.text().strip())
            min_value = float(self.threshold_min_input.text().strip())
        except ValueError:
            QMessageBox.warning(self, "警告", "请输入有效的数值")
            return
        
        if max_value <= min_value:
            QMessageBox.warning(self, "警告", "上限值必须大于下限值")
            return
        
        enabled = self.threshold_enable_check.isChecked()
        remark = self.threshold_remark_input.text().strip()
        self.log_manager.add_threshold_config(config_id, min_value, max_value, enabled, remark)
        
        QMessageBox.information(self, "提示", f"目标序号 {config_id} 的预警阈值已添加")
        self.on_clear_threshold_input()
        self.load_threshold_configs()
    
    def on_update_threshold(self):
        """修改预警阈值"""
        config_id = self.threshold_id_input.text().strip()
        if not config_id:
            QMessageBox.warning(self, "警告", "请输入目标序号")
            return
        
        config = self.log_manager.get_threshold_config(config_id)
        if not config:
            QMessageBox.warning(self, "警告", f"目标序号 {config_id} 的配置不存在，请先添加")
            return
        
        valid_ids = self._get_valid_target_ids()
        if valid_ids and config_id not in valid_ids:
            ids_str = "、".join(valid_ids)
            QMessageBox.warning(
                self, "警告",
                f"目标序号 {config_id} 超出当前范围。\n当前有效序号为：{ids_str}"
            )
            return
        
        try:
            max_value = float(self.threshold_max_input.text().strip())
            min_value = float(self.threshold_min_input.text().strip())
        except ValueError:
            QMessageBox.warning(self, "警告", "请输入有效的数值")
            return
        
        if max_value <= min_value:
            QMessageBox.warning(self, "警告", "上限值必须大于下限值")
            return
        
        enabled = self.threshold_enable_check.isChecked()
        remark = self.threshold_remark_input.text().strip()
        self.log_manager.add_threshold_config(config_id, min_value, max_value, enabled, remark)
        
        QMessageBox.information(self, "提示", f"目标序号 {config_id} 的预警阈值已更新")
        self.on_clear_threshold_input()
        self.load_threshold_configs()
    
    def on_delete_threshold(self):
        """删除预警阈值"""
        config_id = self.threshold_id_input.text().strip()
        if not config_id:
            QMessageBox.warning(self, "警告", "请输入配置ID")
            return
        
        config = self.log_manager.get_threshold_config(config_id)
        if not config:
            QMessageBox.warning(self, "警告", f"配置ID {config_id} 不存在")
            return
        
        reply = QMessageBox.question(self, "确认", f"确定要删除配置ID {config_id} 的预警阈值配置吗？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.log_manager.remove_threshold_config(config_id)
            QMessageBox.information(self, "提示", f"配置ID {config_id} 的预警阈值已删除")
            self.on_clear_threshold_input()
            self.load_threshold_configs()
    
    def on_clear_threshold_input(self):
        """清空阈值输入框"""
        self.threshold_id_input.clear()
        self.threshold_max_input.clear()
        self.threshold_min_input.clear()
        self.threshold_remark_input.clear()
        self.threshold_enable_check.setChecked(True)
    
    def on_edit_threshold(self, config: ThresholdConfig):
        """编辑预警阈值（填充到输入框）"""
        self.threshold_id_input.setText(config.id)
        self.threshold_max_input.setText(str(config.max_value))
        self.threshold_min_input.setText(str(config.min_value))
        self.threshold_remark_input.setText(config.remark)
        self.threshold_enable_check.setChecked(config.enabled)
    
    def on_threshold_table_double_clicked(self, item):
        """双击阈值表格行"""
        row = item.row()
        config_id = self.threshold_table.item(row, 0).text()
        config = self.log_manager.get_threshold_config(config_id)
        if config:
            self.on_edit_threshold(config)
    
    def on_export_monitoring_selected(self):
        """导出选中的监控记录"""
        selected_rows = set([item.row() for item in self.monitoring_table.selectedItems()])
        if not selected_rows:
            QMessageBox.warning(self, "警告", "请先选择要导出的记录")
            return
        
        # TODO: 实现导出功能
        QMessageBox.information(self, "提示", f"将导出 {len(selected_rows)} 条记录（功能待实现）")
    
    def on_export_monitoring_all(self):
        """导出全部监控记录"""
        row_count = self.monitoring_table.rowCount()
        if row_count == 0:
            QMessageBox.warning(self, "警告", "没有可导出的记录")
            return
        
        # TODO: 实现导出功能
        QMessageBox.information(self, "提示", f"将导出 {row_count} 条记录（功能待实现）")
    
    def on_export_alarm_selected(self):
        """导出选中的报警记录"""
        selected_rows = set([item.row() for item in self.alarm_table.selectedItems()])
        if not selected_rows:
            QMessageBox.warning(self, "警告", "请先选择要导出的记录")
            return
        
        # TODO: 实现导出功能
        QMessageBox.information(self, "提示", f"将导出 {len(selected_rows)} 条记录（功能待实现）")
    
    def on_export_alarm_all(self):
        """导出全部报警记录"""
        row_count = self.alarm_table.rowCount()
        if row_count == 0:
            QMessageBox.warning(self, "警告", "没有可导出的记录")
            return
        
        # TODO: 实现导出功能
        QMessageBox.information(self, "提示", f"将导出 {row_count} 条记录（功能待实现）")
    
    def on_clear_monitoring_records(self):
        """清空监控记录"""
        reply = QMessageBox.question(self, "确认", "确定要清空所有监控记录吗？此操作不可恢复！",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.log_manager.clear_monitoring_records()
            self.monitoring_table.setRowCount(0)
            QMessageBox.information(self, "提示", "监控记录已清空")
    
    def on_clear_alarm_records(self):
        """清空报警记录"""
        reply = QMessageBox.question(self, "确认", "确定要清空所有报警记录吗？此操作不可恢复！",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.log_manager.clear_alarm_records()
            self.alarm_table.setRowCount(0)
            QMessageBox.information(self, "提示", "报警记录已清空")
    
    def on_view_record_detail(self, record: MonitoringRecord):
        """查看监控记录详情"""
        detail_text = f"""
记录ID: {record.id}
时间: {record.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
识别值: {record.ocr_value:.2f}
识别文本: {record.ocr_text}
状态: {record.status}
阈值范围: {record.threshold_min if record.threshold_min else 'N/A'} - {record.threshold_max if record.threshold_max else 'N/A'}
        """
        QMessageBox.information(self, "记录详情", detail_text.strip())
    
    def on_view_alarm_detail(self, alarm: AlarmRecord):
        """查看报警记录详情"""
        detail_text = f"""
报警ID: {alarm.id}
时间: {alarm.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
识别值: {alarm.ocr_value:.2f}
阈值范围: {alarm.threshold_min:.2f} - {alarm.threshold_max:.2f}
报警级别: {'超出上限' if alarm.alarm_type == 'above_max' else '低于下限'}
处理状态: {'已处理' if alarm.processed else '未处理'}
        """
        QMessageBox.information(self, "报警详情", detail_text.strip())
    
    def on_mark_alarm_processed(self, alarm: AlarmRecord):
        """标记报警为已处理"""
        self.log_manager.mark_alarm_processed(alarm.id)
        self.load_recent_alarms()
        QMessageBox.information(self, "提示", "报警已标记为已处理")
    
    def on_record_added(self, record: MonitoringRecord):
        """新记录添加信号响应"""
        # 如果当前在监控记录标签页，自动刷新
        if self.tab_widget.currentIndex() == 0:
            self.load_recent_records()
    
    def on_alarm_triggered(self, alarm: AlarmRecord):
        """报警触发信号响应"""
        # 如果当前在报警记录标签页，自动刷新
        if self.tab_widget.currentIndex() == 3:  # 报警记录
            self.load_recent_alarms()
        # 同一区域同类型报警 60 秒内只弹窗一次
        key = (alarm.area_name, alarm.alarm_type)
        now = datetime.now()
        last = self._alarm_popup_cooldown.get(key)
        if last and (now - last).total_seconds() < 60:
            return
        self._alarm_popup_cooldown[key] = now
        QMessageBox.warning(self, "报警", 
                           f"区域 {alarm.area_name} 触发报警！\n"
                           f"识别值: {alarm.ocr_value:.2f}\n"
                           f"阈值范围: {alarm.threshold_min:.2f} - {alarm.threshold_max:.2f}")
    
    def on_threshold_changed(self, area_name: str):
        """阈值配置变更信号响应"""
        self.load_threshold_configs()

