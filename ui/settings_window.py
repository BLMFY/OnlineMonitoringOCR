"""
设置窗口
包含系统信息、相机设置、硬件设备、模型配置等功能
"""
import platform
import os
import shutil
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
                              QGroupBox, QPushButton, QLabel, QComboBox, QLineEdit,
                              QSlider, QSpinBox, QFileDialog, QMessageBox, QFormLayout,
                              QCheckBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from datetime import datetime

from core.settings_manager import SettingsManager
from core.device_manager import DeviceManager
from core.model_manager import ModelManager


class SettingsWindow(QWidget):
    """设置窗口类"""
    
    # 信号定义
    settings_saved = pyqtSignal()  # 设置保存信号
    camera_changed = pyqtSignal(int)  # 相机变更信号
    
    def __init__(self, settings_manager: SettingsManager, 
                 device_manager: DeviceManager, 
                 model_manager: ModelManager,
                 parent=None):
        super().__init__(parent)
        self.settings_manager = settings_manager
        self.device_manager = device_manager
        self.model_manager = model_manager
        
        self.init_ui()
        self.load_settings()
        self.connect_signals()
    
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        self.tab_widget.setFont(QFont("Microsoft YaHei", 10))
        
        # 系统信息标签页
        self.system_tab = self.create_system_tab()
        self.tab_widget.addTab(self.system_tab, "系统信息")
        
        # 相机设置标签页
        self.camera_tab = self.create_camera_tab()
        self.tab_widget.addTab(self.camera_tab, "相机设置")
        
        # 硬件设备标签页
        self.hardware_tab = self.create_hardware_tab()
        self.tab_widget.addTab(self.hardware_tab, "硬件设备")
        
        # 模型配置标签页
        self.model_tab = self.create_model_tab()
        self.tab_widget.addTab(self.model_tab, "模型配置")
        
        layout.addWidget(self.tab_widget)
        
        # 底部按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        btn_save = QPushButton("保存设置")
        btn_save.setFont(QFont("Microsoft YaHei", 10))
        btn_save.setMinimumWidth(100)
        btn_save.setMinimumHeight(35)
        btn_save.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 20px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        btn_save.clicked.connect(self.on_save_settings)
        btn_layout.addWidget(btn_save)
        
        btn_reset = QPushButton("恢复默认")
        btn_reset.setFont(QFont("Microsoft YaHei", 10))
        btn_reset.setMinimumWidth(100)
        btn_reset.setMinimumHeight(35)
        btn_reset.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 20px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
        btn_reset.clicked.connect(self.on_reset_default)
        btn_layout.addWidget(btn_reset)
        
        layout.addLayout(btn_layout)
    
    def create_system_tab(self):
        """创建系统信息标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 系统信息
        info_group = QGroupBox("系统信息")
        info_group.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        info_layout = QFormLayout(info_group)
        info_layout.setSpacing(10)
        
        # 系统版本
        self.system_version_label = QLabel("1.0.0")
        self.system_version_label.setFont(QFont("Microsoft YaHei", 9))
        info_layout.addRow("系统版本:", self.system_version_label)
        
        # 运行时间
        self.uptime_label = QLabel("00:00:00")
        self.uptime_label.setFont(QFont("Microsoft YaHei", 9))
        info_layout.addRow("运行时间:", self.uptime_label)
        
        # 系统状态
        self.system_status_label = QLabel("正常")
        self.system_status_label.setFont(QFont("Microsoft YaHei", 9))
        self.system_status_label.setStyleSheet("color: #27ae60;")
        info_layout.addRow("系统状态:", self.system_status_label)
        
        layout.addWidget(info_group)
        
        # 存储信息
        storage_group = QGroupBox("存储信息")
        storage_group.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        storage_layout = QVBoxLayout(storage_group)
        storage_layout.setSpacing(10)
        
        # 日志存储路径
        log_layout = QHBoxLayout()
        log_layout.addWidget(QLabel("日志存储路径:"))
        self.log_path_label = QLabel(self.settings_manager.config.log_path)
        self.log_path_label.setFont(QFont("Microsoft YaHei", 9))
        self.log_path_label.setStyleSheet("background-color: #ecf0f1; padding: 5px; border-radius: 3px;")
        log_layout.addWidget(self.log_path_label, stretch=1)
        btn_browse_log = QPushButton("浏览")
        btn_browse_log.setFont(QFont("Microsoft YaHei", 9))
        btn_browse_log.setMinimumWidth(80)
        btn_browse_log.clicked.connect(self.on_browse_log_path)
        log_layout.addWidget(btn_browse_log)
        storage_layout.addLayout(log_layout)
        
        # 数据存储路径
        data_layout = QHBoxLayout()
        data_layout.addWidget(QLabel("数据存储路径:"))
        self.data_path_label = QLabel(self.settings_manager.config.data_path)
        self.data_path_label.setFont(QFont("Microsoft YaHei", 9))
        self.data_path_label.setStyleSheet("background-color: #ecf0f1; padding: 5px; border-radius: 3px;")
        data_layout.addWidget(self.data_path_label, stretch=1)
        btn_browse_data = QPushButton("浏览")
        btn_browse_data.setFont(QFont("Microsoft YaHei", 9))
        btn_browse_data.setMinimumWidth(80)
        btn_browse_data.clicked.connect(self.on_browse_data_path)
        data_layout.addWidget(btn_browse_data)
        storage_layout.addLayout(data_layout)
        
        # 已用空间
        self.used_space_label = QLabel("计算中...")
        self.used_space_label.setFont(QFont("Microsoft YaHei", 9))
        storage_layout.addWidget(self.used_space_label)
        
        layout.addWidget(storage_group)
        
        # 刷新按钮
        btn_refresh = QPushButton("刷新信息")
        btn_refresh.setFont(QFont("Microsoft YaHei", 9))
        btn_refresh.clicked.connect(self.refresh_system_info)
        layout.addWidget(btn_refresh)
        
        layout.addStretch()
        
        return widget
    
    def create_camera_tab(self):
        """创建相机设置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 相机设备选择
        device_group = QGroupBox("相机设备选择")
        device_group.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        device_layout = QVBoxLayout(device_group)
        device_layout.setSpacing(10)
        
        # 可用相机
        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("可用相机:"))
        self.camera_combo = QComboBox()
        self.camera_combo.setFont(QFont("Microsoft YaHei", 9))
        self.camera_combo.setMinimumWidth(200)
        camera_layout.addWidget(self.camera_combo)
        
        btn_refresh_camera = QPushButton("刷新设备")
        btn_refresh_camera.setFont(QFont("Microsoft YaHei", 9))
        btn_refresh_camera.setMinimumWidth(100)
        btn_refresh_camera.clicked.connect(self.on_refresh_cameras)
        camera_layout.addWidget(btn_refresh_camera)
        device_layout.addLayout(camera_layout)
        
        # 当前选择
        self.current_camera_label = QLabel("未选择")
        self.current_camera_label.setFont(QFont("Microsoft YaHei", 9))
        self.current_camera_label.setStyleSheet("color: #7f8c8d;")
        device_layout.addWidget(self.current_camera_label)
        
        # 相机信息
        self.camera_info_label = QLabel("")
        self.camera_info_label.setFont(QFont("Microsoft YaHei", 9))
        self.camera_info_label.setStyleSheet("background-color: #ecf0f1; padding: 8px; border-radius: 3px;")
        device_layout.addWidget(self.camera_info_label)
        
        # 测试连接按钮
        btn_test = QPushButton("测试连接")
        btn_test.setFont(QFont("Microsoft YaHei", 9))
        btn_test.setMinimumWidth(100)
        btn_test.clicked.connect(self.on_test_camera)
        device_layout.addWidget(btn_test)
        
        layout.addWidget(device_group)
        
        # 测试模式
        test_group = QGroupBox("测试模式")
        test_group.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        test_layout = QVBoxLayout(test_group)
        test_layout.setSpacing(10)
        
        self.test_mode_check = QCheckBox("启用测试模式")
        self.test_mode_check.setChecked(False)
        self.test_mode_check.setFont(QFont("Microsoft YaHei", 9))
        self.test_mode_check.setToolTip("启用后，主界面开启相机时将显示选定的测试素材（图像或视频）")
        test_layout.addWidget(self.test_mode_check)
        
        material_layout = QHBoxLayout()
        material_layout.addWidget(QLabel("测试素材:"))
        self.test_material_input = QLineEdit()
        self.test_material_input.setPlaceholderText("请选择图像或视频文件")
        self.test_material_input.setReadOnly(True)
        self.test_material_input.setFont(QFont("Microsoft YaHei", 9))
        self.test_material_input.setStyleSheet("background-color: #ecf0f1; padding: 5px; border-radius: 3px;")
        material_layout.addWidget(self.test_material_input, stretch=1)
        btn_browse_material = QPushButton("选择素材")
        btn_browse_material.setFont(QFont("Microsoft YaHei", 9))
        btn_browse_material.setMinimumWidth(100)
        btn_browse_material.clicked.connect(self.on_browse_test_material)
        material_layout.addWidget(btn_browse_material)
        test_layout.addLayout(material_layout)
        
        layout.addWidget(test_group)
        
        # 相机参数配置
        param_group = QGroupBox("相机参数配置")
        param_group.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        param_layout = QFormLayout(param_group)
        param_layout.setSpacing(15)
        
        # 分辨率
        resolution_layout = QHBoxLayout()
        self.camera_width_spin = QSpinBox()
        self.camera_width_spin.setRange(320, 3840)
        self.camera_width_spin.setValue(1280)
        self.camera_width_spin.setFont(QFont("Microsoft YaHei", 9))
        resolution_layout.addWidget(self.camera_width_spin)
        resolution_layout.addWidget(QLabel("x"))
        self.camera_height_spin = QSpinBox()
        self.camera_height_spin.setRange(240, 2160)
        self.camera_height_spin.setValue(720)
        self.camera_height_spin.setFont(QFont("Microsoft YaHei", 9))
        resolution_layout.addWidget(self.camera_height_spin)
        resolution_layout.addStretch()
        param_layout.addRow("分辨率:", resolution_layout)
        
        # 亮度
        brightness_layout = QHBoxLayout()
        self.camera_brightness_slider = QSlider(Qt.Horizontal)
        self.camera_brightness_slider.setRange(0, 100)
        self.camera_brightness_slider.setValue(50)
        self.camera_brightness_slider.valueChanged.connect(
            lambda v: self.camera_brightness_label.setText(f"{v}%"))
        brightness_layout.addWidget(self.camera_brightness_slider)
        self.camera_brightness_label = QLabel("50%")
        self.camera_brightness_label.setMinimumWidth(50)
        self.camera_brightness_label.setFont(QFont("Microsoft YaHei", 9))
        brightness_layout.addWidget(self.camera_brightness_label)
        param_layout.addRow("亮度:", brightness_layout)
        
        # 饱和度
        saturation_layout = QHBoxLayout()
        self.camera_saturation_slider = QSlider(Qt.Horizontal)
        self.camera_saturation_slider.setRange(0, 100)
        self.camera_saturation_slider.setValue(50)
        self.camera_saturation_slider.valueChanged.connect(
            lambda v: self.camera_saturation_label.setText(f"{v}%"))
        saturation_layout.addWidget(self.camera_saturation_slider)
        self.camera_saturation_label = QLabel("50%")
        self.camera_saturation_label.setMinimumWidth(50)
        self.camera_saturation_label.setFont(QFont("Microsoft YaHei", 9))
        saturation_layout.addWidget(self.camera_saturation_label)
        param_layout.addRow("饱和度:", saturation_layout)
        
        layout.addWidget(param_group)
        
        layout.addStretch()
        
        return widget
    
    def create_hardware_tab(self):
        """创建硬件设备标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 报警灯设置
        alarm_group = QGroupBox("报警灯设置")
        alarm_group.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        alarm_layout = QFormLayout(alarm_group)
        alarm_layout.setSpacing(15)
        
        # 报警灯类型
        self.alarm_light_type_combo = QComboBox()
        self.alarm_light_type_combo.addItems(["串口", "USB", "网络"])
        self.alarm_light_type_combo.setFont(QFont("Microsoft YaHei", 9))
        self.alarm_light_type_combo.currentTextChanged.connect(self.on_alarm_light_type_changed)
        alarm_layout.addRow("报警灯类型:", self.alarm_light_type_combo)
        
        # 连接地址
        self.alarm_light_address_input = QLineEdit()
        self.alarm_light_address_input.setPlaceholderText("请输入连接地址")
        self.alarm_light_address_input.setFont(QFont("Microsoft YaHei", 9))
        alarm_layout.addRow("连接地址:", self.alarm_light_address_input)
        
        # 端口号
        self.alarm_light_port_spin = QSpinBox()
        self.alarm_light_port_spin.setRange(1, 65535)
        self.alarm_light_port_spin.setValue(9600)
        self.alarm_light_port_spin.setFont(QFont("Microsoft YaHei", 9))
        alarm_layout.addRow("端口号:", self.alarm_light_port_spin)
        
        # 报警模式
        self.alarm_light_mode_combo = QComboBox()
        self.alarm_light_mode_combo.addItems(["常亮", "闪烁", "声音"])
        self.alarm_light_mode_combo.setFont(QFont("Microsoft YaHei", 9))
        alarm_layout.addRow("报警模式:", self.alarm_light_mode_combo)
        
        # 闪烁频率
        flash_layout = QHBoxLayout()
        self.alarm_light_flash_slider = QSlider(Qt.Horizontal)
        self.alarm_light_flash_slider.setRange(1, 10)
        self.alarm_light_flash_slider.setValue(2)
        self.alarm_light_flash_slider.valueChanged.connect(
            lambda v: self.alarm_light_flash_label.setText(f"{v} 次/秒"))
        flash_layout.addWidget(self.alarm_light_flash_slider)
        self.alarm_light_flash_label = QLabel("2 次/秒")
        self.alarm_light_flash_label.setMinimumWidth(80)
        self.alarm_light_flash_label.setFont(QFont("Microsoft YaHei", 9))
        flash_layout.addWidget(self.alarm_light_flash_label)
        alarm_layout.addRow("闪烁频率:", flash_layout)
        
        # 测试连接按钮
        btn_test_alarm = QPushButton("测试连接")
        btn_test_alarm.setFont(QFont("Microsoft YaHei", 9))
        btn_test_alarm.clicked.connect(self.on_test_alarm_light)
        alarm_layout.addRow("", btn_test_alarm)
        
        layout.addWidget(alarm_group)
        
        layout.addStretch()
        
        return widget
    
    def create_model_tab(self):
        """创建模型配置标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 文字检测模型
        text_detection_group = QGroupBox("文字检测模型")
        text_detection_group.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        text_detection_layout = QFormLayout(text_detection_group)
        text_detection_layout.setSpacing(15)
        
        # 模型选择
        self.text_detection_model_combo = QComboBox()
        self.text_detection_model_combo.setFont(QFont("Microsoft YaHei", 9))
        text_detection_layout.addRow("模型选择:", self.text_detection_model_combo)
        
        # 置信度阈值
        confidence_layout = QHBoxLayout()
        self.text_detection_confidence_slider = QSlider(Qt.Horizontal)
        self.text_detection_confidence_slider.setRange(0, 100)
        self.text_detection_confidence_slider.setValue(50)
        self.text_detection_confidence_slider.valueChanged.connect(
            lambda v: self.text_detection_confidence_label.setText(f"{v/100:.2f}"))
        confidence_layout.addWidget(self.text_detection_confidence_slider)
        self.text_detection_confidence_label = QLabel("0.50")
        self.text_detection_confidence_label.setMinimumWidth(50)
        self.text_detection_confidence_label.setFont(QFont("Microsoft YaHei", 9))
        confidence_layout.addWidget(self.text_detection_confidence_label)
        text_detection_layout.addRow("置信度阈值:", confidence_layout)
        
        layout.addWidget(text_detection_group)
        
        # OCR识别模型
        ocr_group = QGroupBox("OCR识别模型")
        ocr_group.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        ocr_layout = QFormLayout(ocr_group)
        ocr_layout.setSpacing(15)
        
        # 模型选择
        self.ocr_model_combo = QComboBox()
        self.ocr_model_combo.setFont(QFont("Microsoft YaHei", 9))
        ocr_layout.addRow("模型选择:", self.ocr_model_combo)
        
        layout.addWidget(ocr_group)
        
        # 图像增强模型
        enhancement_group = QGroupBox("图像增强模型")
        enhancement_group.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        enhancement_layout = QFormLayout(enhancement_group)
        enhancement_layout.setSpacing(15)
        
        # 模型选择
        self.enhancement_model_combo = QComboBox()
        self.enhancement_model_combo.setFont(QFont("Microsoft YaHei", 9))
        enhancement_layout.addRow("模型选择:", self.enhancement_model_combo)
        
        # 增强强度
        self.enhancement_strength_combo = QComboBox()
        self.enhancement_strength_combo.addItems(["中", "强"])
        self.enhancement_strength_combo.setFont(QFont("Microsoft YaHei", 9))
        enhancement_layout.addRow("增强强度:", self.enhancement_strength_combo)
        
        layout.addWidget(enhancement_group)
        
        layout.addStretch()
        
        return widget
    
    def connect_signals(self):
        """连接信号和槽"""
        self.camera_combo.currentIndexChanged.connect(self.on_camera_selected)
        self.settings_manager.config_changed.connect(self.on_config_changed)
    
    def load_settings(self):
        """加载设置"""
        config = self.settings_manager.get_config()
        
        # 加载相机设置
        self.camera_width_spin.setValue(config.camera_width)
        self.camera_height_spin.setValue(config.camera_height)
        self.test_mode_check.setChecked(getattr(config, 'test_mode_enabled', False))
        self.test_material_input.setText(getattr(config, 'test_material_path', '') or '')
        self.camera_brightness_slider.setValue(int(config.camera_brightness * 100))
        self.camera_saturation_slider.setValue(int(config.camera_saturation * 100))
        
        # 加载硬件设置
        type_map = {"serial": "串口", "usb": "USB", "network": "网络"}
        self.alarm_light_type_combo.setCurrentText(type_map.get(config.alarm_light_type, "串口"))
        self.alarm_light_address_input.setText(config.alarm_light_address)
        self.alarm_light_port_spin.setValue(config.alarm_light_port)
        mode_map = {"always": "常亮", "flash": "闪烁", "sound": "声音"}
        self.alarm_light_mode_combo.setCurrentText(mode_map.get(config.alarm_light_mode, "闪烁"))
        self.alarm_light_flash_slider.setValue(config.alarm_light_flash_frequency)
        
        # 加载模型设置
        self.load_model_combos()
        
        # 设置文字检测模型
        for i in range(self.text_detection_model_combo.count()):
            if self.text_detection_model_combo.itemData(i) == config.text_detection_model:
                self.text_detection_model_combo.setCurrentIndex(i)
                break
        self.text_detection_confidence_slider.setValue(int(config.text_detection_confidence * 100))
        
        # 设置OCR模型
        for i in range(self.ocr_model_combo.count()):
            if self.ocr_model_combo.itemData(i) == config.ocr_model:
                self.ocr_model_combo.setCurrentIndex(i)
                break
        
        # 设置增强模型
        for i in range(self.enhancement_model_combo.count()):
            if self.enhancement_model_combo.itemData(i) == config.enhancement_model:
                self.enhancement_model_combo.setCurrentIndex(i)
                break
        strength_map = {"medium": "中", "strong": "强"}
        self.enhancement_strength_combo.setCurrentText(strength_map.get(config.enhancement_strength, "中"))
        
        # 加载存储路径（做保护处理）
        log_path = config.log_path or "./logs"
        data_path = config.data_path or "./data"
        if not isinstance(log_path, str):
            log_path = "./logs"
        if not isinstance(data_path, str):
            data_path = "./data"
        self.log_path_label.setText(log_path)
        self.data_path_label.setText(data_path)
        
        # 刷新相机列表
        self.on_refresh_cameras()
        
        # 刷新系统信息
        self.refresh_system_info()
    
    def load_model_combos(self):
        """加载模型下拉框"""
        # 文字检测模型
        self.text_detection_model_combo.clear()
        for model_id, model_name in self.settings_manager.get_text_detection_models().items():
            self.text_detection_model_combo.addItem(model_name, model_id)
        
        # OCR识别模型
        self.ocr_model_combo.clear()
        for model_id, model_name in self.settings_manager.get_ocr_models().items():
            self.ocr_model_combo.addItem(model_name, model_id)
        
        # 图像增强模型
        self.enhancement_model_combo.clear()
        for model_id, model_name in self.settings_manager.get_enhancement_models().items():
            self.enhancement_model_combo.addItem(model_name, model_id)
    
    def refresh_system_info(self):
        """刷新系统信息"""
        # 计算运行时间（这里使用启动时间，实际应该从程序启动时记录）
        # TODO: 实现真实的运行时间计算
        self.uptime_label.setText("00:00:00")
        
        # 计算存储空间
        self.update_storage_info()
    
    def update_storage_info(self):
        """更新存储信息"""
        try:
            # 对路径做保护，防止为空
            log_path = self.settings_manager.config.log_path or "./logs"
            data_path = self.settings_manager.config.data_path or "./data"
            
            # 确保路径是字符串类型
            if not isinstance(log_path, str):
                log_path = "./logs"
            if not isinstance(data_path, str):
                data_path = "./data"
            
            total_size = 0
            if os.path.exists(log_path):
                try:
                    for dirpath, dirnames, filenames in os.walk(log_path):
                        for filename in filenames:
                            filepath = os.path.join(dirpath, filename)
                            if os.path.exists(filepath):
                                total_size += os.path.getsize(filepath)
                except Exception as e:
                    print(f"[设置窗口] 计算日志路径空间失败: {e}")
            
            if os.path.exists(data_path):
                try:
                    for dirpath, dirnames, filenames in os.walk(data_path):
                        for filename in filenames:
                            filepath = os.path.join(dirpath, filename)
                            if os.path.exists(filepath):
                                total_size += os.path.getsize(filepath)
                except Exception as e:
                    print(f"[设置窗口] 计算数据路径空间失败: {e}")
            
            # 转换为MB
            size_mb = total_size / (1024 * 1024)
            self.used_space_label.setText(f"已用空间: {size_mb:.2f} MB")
        except Exception as e:
            self.used_space_label.setText(f"计算存储空间失败: {str(e)}")
    
    def on_refresh_cameras(self):
        """刷新相机列表"""
        self.camera_combo.clear()
        cameras = self.device_manager.detect_cameras()
        
        if not cameras:
            self.camera_combo.addItem("未检测到相机")
            self.camera_info_label.setText("未检测到可用相机设备")
            return
        
        for camera in cameras:
            self.camera_combo.addItem(f"{camera['name']} - {camera['info']}", camera['id'])
        
        # 设置当前选择的相机
        current_camera_id = self.settings_manager.config.camera_id
        for i in range(self.camera_combo.count()):
            if self.camera_combo.itemData(i) == current_camera_id:
                self.camera_combo.setCurrentIndex(i)
                break
    
    def on_camera_selected(self, index):
        """相机选择变化"""
        if index < 0:
            return
        
        camera_id = self.camera_combo.itemData(index)
        if camera_id is not None:
            self.current_camera_label.setText(f"当前选择: 相机 {camera_id}")
            info = self.device_manager.get_camera_info(camera_id)
            if info:
                self.camera_info_label.setText(
                    f"分辨率: {info['resolution']}\n"
                    f"状态: {info['status']}"
                )
            else:
                self.camera_info_label.setText("无法获取相机信息")
    
    def on_test_camera(self):
        """测试相机连接"""
        index = self.camera_combo.currentIndex()
        if index < 0:
            QMessageBox.warning(self, "警告", "请先选择相机")
            return
        
        camera_id = self.camera_combo.itemData(index)
        if camera_id is None:
            QMessageBox.warning(self, "警告", "无效的相机选择")
            return
        
        success, message = self.device_manager.test_camera_connection(camera_id)
        if success:
            QMessageBox.information(self, "成功", message)
        else:
            QMessageBox.warning(self, "失败", message)
    
    def on_alarm_light_type_changed(self, text):
        """报警灯类型变化"""
        # 根据类型显示/隐藏相关控件
        if text == "网络":
            self.alarm_light_port_spin.setEnabled(True)
        else:
            self.alarm_light_port_spin.setEnabled(text == "串口")
    
    def on_test_alarm_light(self):
        """测试报警灯连接"""
        type_map = {"串口": "serial", "USB": "usb", "网络": "network"}
        light_type = type_map.get(self.alarm_light_type_combo.currentText(), "serial")
        address = self.alarm_light_address_input.text().strip()
        port = self.alarm_light_port_spin.value()
        
        success, message = self.device_manager.test_alarm_light(light_type, address, port)
        if success:
            QMessageBox.information(self, "成功", message)
        else:
            QMessageBox.warning(self, "失败", message)
    
    def on_browse_log_path(self):
        """浏览日志存储路径"""
        current_path = self.settings_manager.config.log_path
        path = QFileDialog.getExistingDirectory(self, "选择日志存储路径", current_path)
        if path:
            self.settings_manager.update_config(log_path=path)
            self.log_path_label.setText(path)
    
    def on_browse_data_path(self):
        """浏览数据存储路径"""
        current_path = self.settings_manager.config.data_path
        path = QFileDialog.getExistingDirectory(self, "选择数据存储路径", current_path)
        if path:
            self.settings_manager.update_config(data_path=path)
            self.data_path_label.setText(path)
    
    def on_browse_test_material(self):
        """选择测试素材（图像或视频文件）"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择测试素材",
            "",
            "媒体文件 (*.jpg *.jpeg *.png *.bmp *.gif *.mp4 *.avi *.mov *.mkv *.wmv);;"
            "图像文件 (*.jpg *.jpeg *.png *.bmp *.gif);;"
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv);;"
            "所有文件 (*.*)"
        )
        if path:
            self.test_material_input.setText(path)
    
    def on_save_settings(self):
        """保存设置"""
        # 保存相机设置
        camera_id = self.camera_combo.itemData(self.camera_combo.currentIndex())
        if camera_id is not None:
            self.settings_manager.update_config(
                camera_id=camera_id,
                camera_width=self.camera_width_spin.value(),
                camera_height=self.camera_height_spin.value(),
                camera_brightness=self.camera_brightness_slider.value() / 100.0,
                camera_saturation=self.camera_saturation_slider.value() / 100.0
            )
        self.settings_manager.update_config(
            test_mode_enabled=self.test_mode_check.isChecked(),
            test_material_path=self.test_material_input.text().strip()
        )
        
        # 保存硬件设置
        type_map = {"串口": "serial", "USB": "usb", "网络": "network"}
        mode_map = {"常亮": "always", "闪烁": "flash", "声音": "sound"}
        self.settings_manager.update_config(
            alarm_light_type=type_map.get(self.alarm_light_type_combo.currentText(), "serial"),
            alarm_light_address=self.alarm_light_address_input.text().strip(),
            alarm_light_port=self.alarm_light_port_spin.value(),
            alarm_light_mode=mode_map.get(self.alarm_light_mode_combo.currentText(), "flash"),
            alarm_light_flash_frequency=self.alarm_light_flash_slider.value()
        )
        
        # 保存模型设置
        text_detection_model_id = self.text_detection_model_combo.currentData()
        ocr_model_id = self.ocr_model_combo.currentData()
        enhancement_model_id = self.enhancement_model_combo.currentData()
        strength_map = {"中": "medium", "强": "strong"}
        
        if text_detection_model_id:
            self.settings_manager.update_config(
                text_detection_model=text_detection_model_id,
                text_detection_confidence=self.text_detection_confidence_slider.value() / 100.0
            )
        
        if ocr_model_id:
            self.settings_manager.update_config(ocr_model=ocr_model_id)
        
        if enhancement_model_id:
            self.settings_manager.update_config(
                enhancement_model=enhancement_model_id,
                enhancement_strength=strength_map.get(self.enhancement_strength_combo.currentText(), "medium")
            )
        
        # 保存到文件
        if self.settings_manager.save_config():
            QMessageBox.information(self, "提示", "设置已保存")
            self.settings_saved.emit()
        else:
            QMessageBox.warning(self, "错误", "保存设置失败")
    
    def on_reset_default(self):
        """恢复默认设置"""
        reply = QMessageBox.question(self, "确认", "确定要恢复默认设置吗？当前设置将被覆盖。",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.settings_manager.reset_to_default()
            self.load_settings()
            QMessageBox.information(self, "提示", "已恢复默认设置")
    
    def on_config_changed(self):
        """配置变更响应"""
        # 可以在这里实现配置变更的实时响应
        pass

