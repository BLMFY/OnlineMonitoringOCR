"""
智能检测系统主窗口
使用 PyQt5 实现的主界面框架
"""
import os
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSizePolicy,
                              QGroupBox, QPushButton, QLabel, QSlider, QComboBox, QStackedWidget,
                              QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView,
                              QFrame, QAbstractItemView)
from PyQt5.QtCore import Qt, QTimer, QEvent
from PyQt5.QtGui import QFont, QPixmap
from datetime import datetime

# 导入自定义控件
from ui.widgets import AspectRatioLabel

# 导入功能模块
from core.camera_manager import CameraManager
from core.ocr_processor import OCRProcessor
from core.enhancement_processor import EnhancementProcessor
from core.user_manager import UserManager
from core.log_manager import LogManager
from core.settings_manager import SettingsManager
from core.device_manager import DeviceManager
from core.model_manager import ModelManager

# 导入界面
from ui.log_window import LogWindow
from ui.settings_window import SettingsWindow


class MainWindow(QMainWindow):
    """智能检测系统主窗口类"""
    
    def __init__(self):
        super().__init__()
        
        # 初始化功能模块
        self.camera_manager = CameraManager(self)
        self.ocr_processor = OCRProcessor(self)
        self.enhancement_processor = EnhancementProcessor(self)
        self.user_manager = UserManager(self)
        self.log_manager = LogManager(self)
        self.settings_manager = SettingsManager(parent=self)
        self.device_manager = DeviceManager(self)
        self.model_manager = ModelManager(self)
        
        # UI状态变量
        self.video_timer = None  # 视频更新定时器
        self.ocr_timer = None  # OCR 监测定时器（0.5s）
        self.main_widget = None  # 主界面部件
        self.log_widget = None  # 日志界面部件
        self.settings_widget = None  # 设置界面部件
        self.stacked_widget = None  # 堆叠窗口部件
        
        # 提示交互状态
        # None | "area" | "coord" | "target"
        self.hint_mode = None
        self.hint_rect_start = None  # (x, y) label 坐标
        self.hint_rect_end = None
        self.hint_points = []  # [(x,y), ...] label 坐标
        self.last_frame = None  # 最近一帧原始图像（numpy）
        self._hint_rect_pressing = False  # 是否正在拖拽矩形
        self._target_pressing = False  # 框选目标时是否按下未松开
        
        # 初始化UI
        self.init_ui()
        self.setup_timer()
        self.connect_signals()
    
    def connect_signals(self):
        """连接功能模块的信号和槽"""
        # 相机管理信号
        self.camera_manager.camera_opened.connect(self.on_camera_opened)
        self.camera_manager.camera_closed.connect(self.on_camera_closed)
        self.camera_manager.error_occurred.connect(self.on_camera_error)
        self.camera_manager.frame_ready.connect(self.on_frame_ready)
        
        # OCR处理信号
        self.ocr_processor.recognition_started.connect(self.on_ocr_recognition_started)
        self.ocr_processor.recognition_stopped.connect(self.on_ocr_recognition_stopped)
        self.ocr_processor.regions_updated.connect(self.on_ocr_regions_updated)
        self.ocr_processor.regions_cleared.connect(self.on_ocr_regions_cleared)
        
        # 增强处理信号
        self.enhancement_processor.enhancement_started.connect(self.on_enhancement_started)
        self.enhancement_processor.enhancement_stopped.connect(self.on_enhancement_stopped)
        
        # 用户管理信号
        self.user_manager.user_logged_in.connect(self.on_user_logged_in)
        self.user_manager.user_logged_out.connect(self.on_user_logged_out)
    
    def closeEvent(self, event):
        """窗口关闭事件，确保释放相机资源"""
        if self.camera_manager.is_opened():
            self.camera_manager.close_camera()
        if self.video_timer is not None:
            self.video_timer.stop()
            self.video_timer = None
        if self.ocr_timer is not None:
            self.ocr_timer.stop()
            self.ocr_timer = None
        event.accept()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("智能检测系统")
        self.setGeometry(100, 100, 1400, 900)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局：垂直布局，包含顶部信息栏和内容区域
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        
        # 顶部信息与界面切换区（始终显示）
        top_bar = self.create_top_bar()
        central_layout.addWidget(top_bar)
        
        # 创建堆叠窗口用于界面切换
        self.stacked_widget = QStackedWidget()
        central_layout.addWidget(self.stacked_widget)
        
        # 创建主界面
        self.main_widget = self.create_main_interface()
        self.stacked_widget.addWidget(self.main_widget)
        
        # 创建日志界面
        self.log_widget = LogWindow(self.log_manager, self)
        self.stacked_widget.addWidget(self.log_widget)
        
        # 创建设置界面
        self.settings_widget = SettingsWindow(
            self.settings_manager, 
            self.device_manager, 
            self.model_manager,
            self
        )
        self.stacked_widget.addWidget(self.settings_widget)
        
        # 默认显示主界面
        self.stacked_widget.setCurrentIndex(0)
        # 初始化状态条显示（主界面创建后）
        if hasattr(self, 'status_camera'):
            self._update_status_bar()
        # 为视频标签安装事件过滤器（区域/坐标提示交互）
        self.video_label.installEventFilter(self)
        # 加载文本检测模型和识别模型
        self._load_det_model()
        self._load_rec_model()
    
    def create_main_interface(self):
        """创建主界面"""
        main_widget = QWidget()
        
        # 主布局：水平布局，分为左右两个区域
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # ========== 左侧区域 ==========
        left_area = self.create_left_area()
        main_layout.addWidget(left_area, stretch=3)
        
        # ========== 右侧区域 ==========
        right_area = self.create_right_area()
        main_layout.addWidget(right_area, stretch=1)
        
        return main_widget
        
    def create_left_area(self):
        """创建左侧区域：实时监控区 + 下方状态条"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(8)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # 实时监控区
        monitor_area = self.create_monitor_area()
        left_layout.addWidget(monitor_area, stretch=1)
        
        # 监视窗口下方状态条
        status_bar = self.create_status_bar()
        left_layout.addWidget(status_bar)
        
        return left_widget
        
    def create_top_bar(self):
        """创建顶部信息与界面切换区"""
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setSpacing(15)
        top_layout.setContentsMargins(10, 10, 10, 10)
        
        # 界面切换区域
        switch_label = QLabel("界面切换:")
        switch_label.setFont(QFont("Microsoft YaHei", 9))
        self.interface_combo = QComboBox()
        self.interface_combo.addItems(["主界面", "设置", "日志"])
        self.interface_combo.setFont(QFont("Microsoft YaHei", 9))
        self.interface_combo.setMinimumWidth(120)
        self.interface_combo.currentIndexChanged.connect(self.on_interface_changed)
        
        top_layout.addWidget(switch_label)
        top_layout.addWidget(self.interface_combo)
        top_layout.addStretch()
        
        # 时间戳显示
        time_label = QLabel("当前时间:")
        time_label.setFont(QFont("Microsoft YaHei", 9))
        self.time_display = QLabel()
        self.time_display.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        self.time_display.setStyleSheet("color: #2c3e50; padding: 5px;")
        self.update_time_display()
        
        top_layout.addWidget(time_label)
        top_layout.addWidget(self.time_display)
        top_layout.addSpacing(20)
        
        # 用户信息显示
        user_label = QLabel("当前用户:")
        user_label.setFont(QFont("Microsoft YaHei", 9))
        current_user = self.user_manager.get_current_user()
        self.user_display = QLabel(current_user if current_user else "未登录")
        self.user_display.setFont(QFont("Microsoft YaHei", 9))
        self.user_display.setStyleSheet("color: #34495e; padding: 5px;")
        
        top_layout.addWidget(user_label)
        top_layout.addWidget(self.user_display)
        top_layout.addSpacing(10)
        
        # 登录/退出按钮
        self.btn_login = QPushButton("登录")
        self.btn_login.setFont(QFont("Microsoft YaHei", 9))
        self.btn_login.setMinimumWidth(80)
        self.btn_login.clicked.connect(self.on_login_clicked)
        
        self.btn_logout = QPushButton("退出")
        self.btn_logout.setFont(QFont("Microsoft YaHei", 9))
        self.btn_logout.setMinimumWidth(80)
        self.btn_logout.setEnabled(False)
        self.btn_logout.clicked.connect(self.on_logout_clicked)
        
        top_layout.addWidget(self.btn_login)
        top_layout.addWidget(self.btn_logout)
        
        return top_widget
        
    def create_monitor_area(self):
        """创建实时监控区"""
        monitor_widget = QWidget()
        monitor_widget.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
            }
        """)
        
        monitor_layout = QVBoxLayout(monitor_widget)
        monitor_layout.setContentsMargins(0, 0, 0, 0)
        
        # 视频显示标签（占位）- 使用固定宽高比，可缩放
        self.video_label = AspectRatioLabel(aspect_ratio=16/9, parent=monitor_widget)
        self.video_label.setText("实时监控区\n（等待相机启动...）")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #2d2d2d;
                color: #888;
                font-size: 16px;
            }
        """)
        # 设置大小策略，允许缩放但保持比例
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        monitor_layout.addWidget(self.video_label, stretch=1)
        
        return monitor_widget
    
    def create_status_bar(self):
        """创建监视窗口下方的系统状态条"""
        bar = QWidget()
        bar.setFixedHeight(32)
        bar.setStyleSheet("""
            QWidget {
                background-color: #2c3e50;
                border-radius: 4px;
            }
        """)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(24)
        
        self.status_camera = QLabel("相机: 关")
        self.status_ocr = QLabel("监测: 关")
        self.status_enhance = QLabel("增强: 关")
        self.status_target_count = QLabel("监测目标数: 0")
        for lb in (self.status_camera, self.status_ocr, self.status_enhance, self.status_target_count):
            lb.setFont(QFont("Microsoft YaHei", 9))
            lb.setStyleSheet("color: #ecf0f1;")
            layout.addWidget(lb)
        layout.addStretch()
        return bar
        
    def create_right_area(self):
        """创建右侧区域：相机/监测/增强开关 + OCR 结果区（无分区标题）"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(12)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # 相机开关 + 变焦（无标题分区）
        camera_block = self.create_camera_block()
        right_layout.addWidget(camera_block)
        
        # 分隔线
        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setStyleSheet("background-color: #bdc3c7; max-height: 1px;")
        right_layout.addWidget(line1)
        
        # OCR 监测开关 + 四个辅助按钮 + 图像增强开关（无标题分区）
        controls_block = self.create_controls_block()
        right_layout.addWidget(controls_block)
        self.right_controls_container = controls_block  # 相机未开时禁用
        self.right_controls_container.setEnabled(False)
        
        # 分隔线
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setStyleSheet("background-color: #bdc3c7; max-height: 1px;")
        right_layout.addWidget(line2)
        
        # OCR 监测结果实时展示区
        ocr_results_block = self.create_ocr_results_block()
        right_layout.addWidget(ocr_results_block, stretch=1)
        
        return right_widget
    
    def _create_switch_slider(self, on_color="#27ae60"):
        """创建两档 QSlider 开关（0=关，1=开），样式为轨道+滑块"""
        s = QSlider(Qt.Horizontal)
        s.setMinimum(0)
        s.setMaximum(1)
        s.setValue(0)
        s.setSingleStep(1)
        s.setPageStep(1)
        s.setFixedWidth(52)
        s.setFixedHeight(28)
        s.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 22px;
                border-radius: 11px;
                background: #bdc3c7;
                border: 1px solid #95a5a6;
            }}
            QSlider::handle:horizontal {{
                width: 18px;
                height: 18px;
                margin: 1px;
                border-radius: 9px;
                background: white;
                border: 1px solid #95a5a6;
            }}
            QSlider::handle:horizontal:hover {{
                background: #f8f8f8;
            }}
            QSlider::sub-page:horizontal {{
                background: {on_color};
                border-radius: 11px;
            }}
        """)
        return s
    
    def create_camera_block(self):
        """相机开关 + 变焦（无 GroupBox 标题）"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 相机开关（QSlider 两档）+ 右侧状态标注
        cam_row = QHBoxLayout()
        cam_label = QLabel("相机")
        cam_label.setFont(QFont("Microsoft YaHei", 10))
        cam_row.addWidget(cam_label)
        self.switch_camera = self._create_switch_slider("#27ae60")
        self.switch_camera.valueChanged.connect(self.on_camera_switch_changed)
        cam_row.addWidget(self.switch_camera)
        self.label_camera_status = QLabel("关闭")
        self.label_camera_status.setFont(QFont("Microsoft YaHei", 9))
        self.label_camera_status.setStyleSheet("color: #7f8c8d; min-width: 36px;")
        cam_row.addWidget(self.label_camera_status)
        cam_row.addStretch()
        layout.addLayout(cam_row)
        
        # 变焦
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("变焦"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setMinimum(0)
        self.zoom_slider.setMaximum(100)
        self.zoom_slider.setValue(50)
        self.zoom_slider.setEnabled(False)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        self.zoom_value_label = QLabel("50")
        self.zoom_value_label.setFont(QFont("Microsoft YaHei", 9))
        self.zoom_value_label.setMinimumWidth(36)
        self.zoom_value_label.setAlignment(Qt.AlignCenter)
        self.zoom_value_label.setStyleSheet("background-color: #ecf0f1; padding: 2px; border-radius: 2px;")
        zoom_layout.addWidget(self.zoom_slider)
        zoom_layout.addWidget(self.zoom_value_label)
        layout.addLayout(zoom_layout)
        
        return widget
    
    def create_controls_block(self):
        """OCR 监测开关 + 四按钮 + 图像增强开关（无 GroupBox 标题）"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # OCR 监测开关（QSlider 两档）+ 右侧状态标注
        ocr_row = QHBoxLayout()
        ocr_label = QLabel("OCR 监测")
        ocr_label.setFont(QFont("Microsoft YaHei", 10))
        ocr_row.addWidget(ocr_label)
        self.switch_ocr = self._create_switch_slider("#3498db")
        self.switch_ocr.valueChanged.connect(self.on_ocr_switch_changed)
        ocr_row.addWidget(self.switch_ocr)
        self.label_ocr_status = QLabel("关闭")
        self.label_ocr_status.setFont(QFont("Microsoft YaHei", 9))
        self.label_ocr_status.setStyleSheet("color: #7f8c8d; min-width: 36px;")
        ocr_row.addWidget(self.label_ocr_status)
        ocr_row.addStretch()
        layout.addLayout(ocr_row)
        
        # 四个辅助按钮：两两一排
        btn_style = """
            QPushButton { background-color: #3498db; color: white; border: none; border-radius: 4px; padding: 6px; font-size: 9pt; }
            QPushButton:hover { background-color: #2980b9; }
            QPushButton:disabled { background-color: #bdc3c7; color: #7f8c8d; }
        """
        row1 = QHBoxLayout()
        self.btn_ocr_select_target = QPushButton("框选目标")
        self.btn_ocr_select_target.setFont(QFont("Microsoft YaHei", 9))
        self.btn_ocr_select_target.setStyleSheet(btn_style)
        self.btn_ocr_select_target.clicked.connect(self.on_ocr_select_target)
        self.btn_ocr_global_clean = QPushButton("刷新提示")
        self.btn_ocr_global_clean.setFont(QFont("Microsoft YaHei", 9))
        self.btn_ocr_global_clean.setStyleSheet(btn_style)
        self.btn_ocr_global_clean.clicked.connect(self.on_ocr_global_clean)
        row1.addWidget(self.btn_ocr_select_target)
        row1.addWidget(self.btn_ocr_global_clean)
        layout.addLayout(row1)
        row2 = QHBoxLayout()
        self.btn_ocr_select_area = QPushButton("区域提示")
        self.btn_ocr_select_area.setFont(QFont("Microsoft YaHei", 9))
        self.btn_ocr_select_area.setStyleSheet(btn_style)
        self.btn_ocr_select_area.clicked.connect(self.on_ocr_select_area)
        self.btn_ocr_click_hint = QPushButton("坐标提示")
        self.btn_ocr_click_hint.setFont(QFont("Microsoft YaHei", 9))
        self.btn_ocr_click_hint.setStyleSheet(btn_style)
        self.btn_ocr_click_hint.clicked.connect(self.on_ocr_click_hint)
        row2.addWidget(self.btn_ocr_select_area)
        row2.addWidget(self.btn_ocr_click_hint)
        layout.addLayout(row2)
        
        # 图像增强开关（QSlider 两档）+ 右侧状态标注
        enhance_row = QHBoxLayout()
        enhance_label = QLabel("图像增强")
        enhance_label.setFont(QFont("Microsoft YaHei", 10))
        enhance_row.addWidget(enhance_label)
        self.switch_enhance = self._create_switch_slider("#e67e22")
        self.switch_enhance.valueChanged.connect(self.on_enhance_switch_changed)
        enhance_row.addWidget(self.switch_enhance)
        self.label_enhance_status = QLabel("关闭")
        self.label_enhance_status.setFont(QFont("Microsoft YaHei", 9))
        self.label_enhance_status.setStyleSheet("color: #7f8c8d; min-width: 36px;")
        enhance_row.addWidget(self.label_enhance_status)
        enhance_row.addStretch()
        layout.addLayout(enhance_row)
        
        return widget
    
    def create_ocr_results_block(self):
        """右侧 OCR 监测结果实时展示区（浅灰色）"""
        widget = QWidget()
        widget.setStyleSheet("background-color: #e8e8e8; border-radius: 4px; padding: 8px;")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        
        title = QLabel("监测结果")
        title.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        title.setStyleSheet("color: #2c3e50; background: transparent;")
        layout.addWidget(title)
        
        self.ocr_results_table = QTableWidget()
        self.ocr_results_table.setColumnCount(2)
        self.ocr_results_table.setHorizontalHeaderLabels(["目标序号", "识别结果"])
        self.ocr_results_table.setFont(QFont("Microsoft YaHei", 9))
        self.ocr_results_table.setAlternatingRowColors(True)
        self.ocr_results_table.horizontalHeader().setStretchLastSection(True)
        self.ocr_results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ocr_results_table.setMinimumHeight(120)
        self.ocr_results_table.setStyleSheet("""
            QTableWidget { background-color: #f5f5f5; border: 1px solid #d0d0d0; border-radius: 3px; }
            QTableWidget::item { background-color: #fafafa; }
            QTableWidget::item:alternate { background-color: #f0f0f0; }
            QHeaderView::section { background-color: #e0e0e0; padding: 4px; }
        """)
        layout.addWidget(self.ocr_results_table)
        
        return widget
    
    def set_ocr_results(self, items):
        """设置监测结果表格数据，items 为 [(目标序号, 识别结果), ...]，并刷新状态条"""
        self.ocr_results_table.setRowCount(len(items))
        for row, (target_id, value) in enumerate(items):
            self.ocr_results_table.setItem(row, 0, QTableWidgetItem(str(target_id)))
            self.ocr_results_table.setItem(row, 1, QTableWidgetItem(str(value)))
        self._update_status_bar()
    
    def _label_to_frame(self, lx: float, ly: float):
        """将 label 坐标转换为帧坐标，若点在显示区域外返回 None"""
        if self.last_frame is None:
            return None
        h, w = self.last_frame.shape[:2]
        lw = self.video_label.width()
        lh = self.video_label.height()
        if lw <= 0 or lh <= 0 or w <= 0 or h <= 0:
            return None
        # 使用拉伸映射（与 setScaledContents 显示的拉伸一致）
        fx = lx * w / lw
        fy = ly * h / lh
        fx = max(0, min(fx, w - 1))
        fy = max(0, min(fy, h - 1))
        return (fx, fy)
    
    def _frame_to_label(self, fx: float, fy: float):
        """将帧坐标转换为 label 坐标"""
        if self.last_frame is None:
            return (0, 0)
        h, w = self.last_frame.shape[:2]
        lw = self.video_label.width()
        lh = self.video_label.height()
        if lw <= 0 or lh <= 0 or w <= 0 or h <= 0:
            return (0, 0)
        # 使用拉伸映射（与 setScaledContents 显示的拉伸一致）
        lx = fx * lw / w
        ly = fy * lh / h
        return (lx, ly)
    
    def _load_det_model(self):
        """从设置加载文本检测模型"""
        cfg = self.settings_manager.get_config()
        model_id = getattr(cfg, 'text_detection_model', 'model_1')
        path = self.model_manager.get_det_model_path(model_id)
        if path:
            self.ocr_processor.load_det_model(path)
        else:
            print(f"[主窗口] 检测模型权重未找到，model_id={model_id}")
    
    def _load_rec_model(self):
        """从设置加载文本识别模型"""
        cfg = self.settings_manager.get_config()
        model_id = getattr(cfg, 'ocr_model', 'ocr_model_1')
        weight_path = self.model_manager.get_ocr_model_path(model_id)
        dict_path = self.model_manager.get_ocr_dict_path(model_id)
        if weight_path and dict_path:
            self.ocr_processor.load_rec_model(weight_path, dict_path)
        else:
            print(f"[主窗口] 识别模型或字典未找到，model_id={model_id}")
    
    def _run_ocr_cycle(self):
        """每 0.5s 执行一次：识别所有目标区域，更新 UI 和日志"""
        if not self.camera_manager.is_opened() or self.last_frame is None:
            return
        if not self.ocr_processor.has_rec_model():
            return
        regions = self.ocr_processor.get_regions()
        if not regions:
            return
        results = self.ocr_processor.recognize_regions(self.last_frame, conf_threshold=0.7)
        if not results:
            return
        items = []
        live_rows = []
        for r in results:
            region_id = str(r["region"].id)
            if r["score"] < 0.7:
                display_text = "/"
            else:
                display_text = r["text"] if r["text"] else "/"
            items.append((region_id, display_text))
            # 计算状态用于监测动态
            num_val = r.get("numeric")
            config = self.log_manager.get_threshold_config_by_area(region_id)
            if config and config.enabled:
                if num_val is not None:
                    if num_val > config.max_value:
                        status = "警告：超出上限"
                    elif num_val < config.min_value:
                        status = "警告：超出下限"
                    else:
                        status = "正常"
                else:
                    status = "正常"
            else:
                status = "正常"
            value_str = f"{num_val:.2f}" if isinstance(num_val, (int, float)) else (str(num_val) if num_val is not None else "/")
            live_rows.append((region_id, value_str, display_text, status, ""))
            if r["score"] >= 0.7 and r["numeric"] is not None:
                self.log_manager.add_monitoring_record(
                    area_name=region_id,
                    ocr_value=r["numeric"],
                    ocr_text=r["text"],
                    remark=""
                )
        self.set_ocr_results(items)
        if hasattr(self, "log_widget") and hasattr(self.log_widget, "update_live_display"):
            self.log_widget.update_live_display(live_rows)
        
    def setup_timer(self):
        """设置定时器用于更新时间显示"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time_display)
        self.timer.start(1000)  # 每秒更新一次
    
    def eventFilter(self, obj, event):
        """处理视频标签的鼠标事件（区域提示、坐标提示）"""
        if obj is not self.video_label or not self.camera_manager.is_opened():
            return super().eventFilter(obj, event)
        
        t = event.type()
        if t == QEvent.MouseButtonPress:
            pos = event.pos()
            x, y = pos.x(), pos.y()
            if event.button() == Qt.LeftButton:
                if self.hint_mode == "area":
                    self._hint_rect_pressing = True
                    self.hint_rect_start = (x, y)
                    self.hint_rect_end = (x, y)
                    self.video_label.set_hint_rect(self.hint_rect_start, self.hint_rect_end)
                elif self.hint_mode == "coord":
                    # 坐标提示：仅记录提示点，不立即生成区域
                    self.hint_points.append((x, y))
                    self.video_label.set_hint_points(self.hint_points)
                elif self.hint_mode == "target":
                    # 框选目标：按下不立即确定，在 MouseButtonRelease 中确认
                    self._target_pressing = True
            elif event.button() == Qt.RightButton:
                if self.hint_mode == "coord":
                    # 坐标提示：右键撤销最后一个点
                    if self.hint_points:
                        self.hint_points.pop()
                        self.video_label.set_hint_points(self.hint_points)
                elif self.hint_mode == "target":
                    # 框选目标：在完成四个点之前右键可取消本次框选
                    self._target_pressing = False
                    self.hint_mode = None
                    self.hint_points = []
                    self.video_label.set_target_points([])
                    self.video_label.setCursor(Qt.ArrowCursor)
            return True
        elif t == QEvent.MouseMove:
            pos = event.pos()
            x, y = pos.x(), pos.y()
            if self.hint_mode == "area" and self._hint_rect_pressing:
                self.hint_rect_end = (x, y)
                self.video_label.set_hint_rect(self.hint_rect_start, self.hint_rect_end)
            elif self.hint_mode == "target" and self._target_pressing:
                # 框选目标：未松开时显示预览点和连线
                self.video_label.set_target_points(self.hint_points, preview=(x, y))
            return True
        elif t == QEvent.MouseButtonRelease:
            if event.button() == Qt.LeftButton and self.hint_mode == "target":
                # 框选目标：松开鼠标后确认点
                self._target_pressing = False
                pos = event.pos()
                x, y = pos.x(), pos.y()
                self.hint_points.append((x, y))
                self.video_label.set_target_points(self.hint_points, preview=None)
                if len(self.hint_points) == 4:
                    # 将 label 坐标转换为帧坐标并添加为手动区域
                    frame_points = []
                    for lx, ly in self.hint_points:
                        pt = self._label_to_frame(lx, ly)
                        if not pt:
                            frame_points = []
                            break
                        frame_points.append(pt)
                        if frame_points and self.last_frame is not None:
                            self.ocr_processor.add_manual_region(frame_points, from_mode="target")
                        # 退出框选模式，清除临时提示点但保留已添加的区域
                        self._target_pressing = False
                        self.hint_mode = None
                        self.hint_points = []
                        self.video_label.set_target_points([])
                        self.video_label.setCursor(Qt.ArrowCursor)
            elif event.button() == Qt.LeftButton and self.hint_mode == "area" and self._hint_rect_pressing:
                self._hint_rect_pressing = False
                pos = event.pos()
                x, y = pos.x(), pos.y()
                self.hint_rect_end = (x, y)
                self.video_label.set_hint_rect(None, None)
                # 转换为帧坐标并检测
                p1 = self._label_to_frame(self.hint_rect_start[0], self.hint_rect_start[1])
                p2 = self._label_to_frame(self.hint_rect_end[0], self.hint_rect_end[1])
                if p1 and p2 and self.last_frame is not None:
                    x1, y1 = p1
                    x2, y2 = p2
                    roi_x = int(min(x1, x2))
                    roi_y = int(min(y1, y2))
                    roi_w = int(max(1, abs(x2 - x1)))
                    roi_h = int(max(1, abs(y2 - y1)))
                    self.ocr_processor.detect_in_roi(self.last_frame, (roi_x, roi_y, roi_w, roi_h))
                self.hint_mode = None
                self.hint_rect_start = None
                self.hint_rect_end = None
                self.video_label.setCursor(Qt.ArrowCursor)
            return True
        return super().eventFilter(obj, event)
        
    def update_time_display(self):
        """更新时间显示"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_display.setText(current_time)
    
    def update_video_frame(self):
        """更新视频帧显示"""
        ret, frame = self.camera_manager.read_frame()
        if not ret:
            return
        
        # 如果启用了增强功能，先处理帧
        if self.enhancement_processor.is_processing():
            frame = self.enhancement_processor.process_frame(frame)
        
        # 保存当前帧用于检测（与用户所见一致）
        self.last_frame = frame.copy()
        
        # 获取视频标签的当前大小
        label_width = self.video_label.width()
        label_height = self.video_label.height()
        target_size = (label_width, label_height) if label_width > 0 and label_height > 0 else None
        
        # 转换为QImage并显示
        qt_image = self.camera_manager.frame_to_qimage(frame, target_size)
        if qt_image is not None:
            pixmap = QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pixmap)
            self.video_label.setAlignment(Qt.AlignCenter)
    
    # ========== 信号槽函数 ==========
    
    def on_interface_changed(self, index):
        """界面切换槽函数"""
        interface_name = self.interface_combo.currentText()
        print(f"[界面切换] 切换到: {interface_name}")
        
        # 根据选择的界面名称切换
        if interface_name == "主界面":
            if self.stacked_widget.currentIndex() != 0:
                self.stacked_widget.setCurrentIndex(0)
        elif interface_name == "日志":
            if self.stacked_widget.currentIndex() != 1:
                self.stacked_widget.setCurrentIndex(1)
        elif interface_name == "设置":
            if self.stacked_widget.currentIndex() != 2:
                self.stacked_widget.setCurrentIndex(2)
        
    def on_login_clicked(self):
        """登录按钮槽函数"""
        # 调用用户管理模块的登录方法
        # 这里使用测试用户，实际应该弹出登录对话框
        success = self.user_manager.login("测试用户")
        if not success:
            print("[登录] 登录失败")
        
    def on_logout_clicked(self):
        """退出按钮槽函数"""
        self.user_manager.logout()
        
    def on_open_camera(self):
        """开启相机槽函数"""
        cfg = self.settings_manager.get_config()
        test_enabled = getattr(cfg, 'test_mode_enabled', False)
        test_path = (getattr(cfg, 'test_material_path', '') or '').strip()
        if test_enabled and test_path:
            success = self.camera_manager.open_camera(
                camera_id=cfg.camera_id,
                test_mode=True,
                test_material_path=test_path
            )
        else:
            success = self.camera_manager.open_camera(cfg.camera_id)
        if success:
            # 启动视频更新定时器
            self.video_timer = QTimer()
            self.video_timer.timeout.connect(self.update_video_frame)
            self.video_timer.start(33)  # 约30fps
        
    def on_close_camera(self):
        """关闭相机槽函数（若监测/增强已开则先关闭）"""
        if self.ocr_timer is not None:
            self.ocr_timer.stop()
        if self.ocr_processor.is_processing():
            self.ocr_processor.stop_recognition()
        if self.enhancement_processor.is_processing():
            self.enhancement_processor.stop_enhancement()
        self.camera_manager.close_camera()
        if self.video_timer is not None:
            self.video_timer.stop()
            self.video_timer = None
    
    def on_camera_switch_changed(self, value):
        """相机开关状态变化（QSlider 0=关 1=开）"""
        if value == 1:
            self.on_open_camera()
        else:
            self.on_close_camera()
    
    def on_ocr_switch_changed(self, value):
        """OCR 监测开关（QSlider 0=关 1=开）"""
        if value == 1:
            self.on_ocr_start()
        else:
            self.on_ocr_stop()
    
    def on_enhance_switch_changed(self, value):
        """图像增强开关（QSlider 0=关 1=开）"""
        if value == 1:
            self.on_enhance_start()
        else:
            self.on_enhance_stop()
    
    def _update_status_bar(self):
        """根据当前状态更新状态条文案及开关右侧标注"""
        cam_on = self.camera_manager.is_opened()
        ocr_on = self.ocr_processor.is_processing()
        enh_on = self.enhancement_processor.is_processing()
        self.status_camera.setText("相机: 开" if cam_on else "相机: 关")
        self.status_ocr.setText("监测: 开" if ocr_on else "监测: 关")
        self.status_enhance.setText("增强: 开" if enh_on else "增强: 关")
        n = self.ocr_results_table.rowCount() if hasattr(self, 'ocr_results_table') else 0
        self.status_target_count.setText(f"监测目标数: {n}")
        # 开关右侧状态标注
        if hasattr(self, 'label_camera_status'):
            self.label_camera_status.setText("开启" if cam_on else "关闭")
            self.label_camera_status.setStyleSheet("color: #27ae60;" if cam_on else "color: #7f8c8d; min-width: 36px;")
        if hasattr(self, 'label_ocr_status'):
            self.label_ocr_status.setText("开启" if ocr_on else "关闭")
            self.label_ocr_status.setStyleSheet("color: #3498db;" if ocr_on else "color: #7f8c8d; min-width: 36px;")
        if hasattr(self, 'label_enhance_status'):
            self.label_enhance_status.setText("开启" if enh_on else "关闭")
            self.label_enhance_status.setStyleSheet("color: #e67e22;" if enh_on else "color: #7f8c8d; min-width: 36px;")
        
    def on_zoom_changed(self, value):
        """镜头变焦槽函数"""
        self.zoom_value_label.setText(str(value))
        self.camera_manager.set_zoom(value)
        
    def on_ocr_select_target(self):
        """框选目标：顺次点击 4 个点形成一个待测文本区域"""
        if not self.camera_manager.is_opened():
            QMessageBox.warning(self, "提示", "请先打开相机")
            return
        if self.last_frame is None:
            QMessageBox.warning(self, "提示", "请等待画面加载完成")
            return
        # 进入框选目标模式，不清除已有区域
        self.hint_mode = "target"
        self.hint_points = []
        self.video_label.set_target_points([])
        self.video_label.setCursor(Qt.CrossCursor)
        
    def on_ocr_select_area(self):
        """区域提示：进入拖拽矩形模式"""
        if not self.camera_manager.is_opened():
            QMessageBox.warning(self, "提示", "请先打开相机")
            return
        if self.last_frame is None:
            QMessageBox.warning(self, "提示", "请等待画面加载完成")
            return
        if not self.ocr_processor.has_det_model():
            QMessageBox.warning(self, "提示", "检测模型未加载，请在设置中配置模型权重路径")
            return
        self.hint_mode = "area"
        self._hint_rect_pressing = False
        self.hint_rect_start = None
        self.hint_rect_end = None
        self.video_label.set_hint_rect(None, None)
        self.video_label.setCursor(Qt.CrossCursor)
        
    def on_ocr_click_hint(self):
        """坐标提示 / 确认：进入坐标模式或确认检测"""
        if not self.camera_manager.is_opened():
            QMessageBox.warning(self, "提示", "请先打开相机")
            return
        if self.last_frame is None:
            QMessageBox.warning(self, "提示", "请等待画面加载完成")
            return
        if not self.ocr_processor.has_det_model():
            QMessageBox.warning(self, "提示", "检测模型未加载，请在设置中配置模型权重路径")
            return
        if self.hint_mode == "coord":
            # 当前在坐标模式，点击确认
            if not self.hint_points:
                QMessageBox.warning(self, "提示", "请先点击画面添加至少一个提示点")
                return
            frame_points = []
            for lx, ly in self.hint_points:
                pt = self._label_to_frame(lx, ly)
                if pt:
                    frame_points.append(pt)
            if frame_points:
                self.ocr_processor.detect_with_points(self.last_frame, frame_points)
            self.hint_mode = None
            self.hint_points = []
            self.video_label.set_hint_points([])
            self.video_label.setCursor(Qt.ArrowCursor)
            self.btn_ocr_click_hint.setText("坐标提示")
        else:
            # 进入坐标提示模式
            self.hint_mode = "coord"
            self.hint_points = []
            self.video_label.set_hint_points([])
            self.video_label.setCursor(Qt.CrossCursor)
            self.btn_ocr_click_hint.setText("确认")
        
    def on_ocr_global_clean(self):
        """刷新提示：清除所有文本区域和覆盖层"""
        self.ocr_processor.clear_regions()
        self.hint_points = []
        self.video_label.clear_overlay()
        # 同步清空右侧 OCR 结果表格与监测目标数
        self.set_ocr_results([])
        # 清空预警设置
        self.log_manager.clear_threshold_configs()
        if hasattr(self, "log_widget") and hasattr(self.log_widget, "load_threshold_configs"):
            self.log_widget.load_threshold_configs()
        if hasattr(self, "log_widget") and hasattr(self.log_widget, "update_live_display"):
            self.log_widget.update_live_display([])
        
    def on_ocr_start(self):
        """启动监测：检查条件后启动 0.5s 定时识别"""
        if not self.camera_manager.is_opened():
            QMessageBox.warning(self, "提示", "请先打开相机")
            self._reset_ocr_switch_off()
            return
        if self.last_frame is None:
            QMessageBox.warning(self, "提示", "请等待画面加载完成")
            self._reset_ocr_switch_off()
            return
        if not self.ocr_processor.has_det_model():
            QMessageBox.warning(self, "提示", "检测模型未加载，请在设置中配置")
            self._reset_ocr_switch_off()
            return
        if not self.ocr_processor.has_rec_model():
            QMessageBox.warning(self, "提示", "识别模型未加载，请在设置中配置")
            self._reset_ocr_switch_off()
            return
        if not self.ocr_processor.get_regions():
            QMessageBox.warning(self, "提示", "请先通过区域提示或坐标提示完成文本区域选择")
            self._reset_ocr_switch_off()
            return
        if self.ocr_timer is None:
            self.ocr_timer = QTimer()
            self.ocr_timer.timeout.connect(self._run_ocr_cycle)
        self.ocr_timer.start(500)
        self.ocr_processor.start_recognition()
        
    def _reset_ocr_switch_off(self):
        """将 OCR 监测开关复位为关闭"""
        if hasattr(self, 'switch_ocr'):
            self.switch_ocr.blockSignals(True)
            self.switch_ocr.setValue(0)
            self.switch_ocr.blockSignals(False)
        self._update_status_bar()
    
    def on_ocr_stop(self):
        """关闭监测：停止定时器并停止识别"""
        if self.ocr_timer is not None:
            self.ocr_timer.stop()
        self.ocr_processor.stop_recognition()
        # 清空右侧结果表格
        self.set_ocr_results([])
        # 清空日志窗口中当前监控记录表格显示（不删除历史数据）
        if hasattr(self, "log_widget") and hasattr(self.log_widget, "monitoring_table"):
            self.log_widget.monitoring_table.setRowCount(0)
        # 清空监测动态
        if hasattr(self, "log_widget") and hasattr(self.log_widget, "update_live_display"):
            self.log_widget.update_live_display([])
        
    def on_enhance_start(self):
        """启动增强槽函数"""
        self.enhancement_processor.start_enhancement()
        
    def on_enhance_stop(self):
        """关闭增强槽函数"""
        self.enhancement_processor.stop_enhancement()
    
    # ========== 功能模块信号响应 ==========
    
    def on_camera_opened(self):
        """相机打开信号响应"""
        self.switch_camera.blockSignals(True)
        self.switch_camera.setValue(1)
        self.switch_camera.blockSignals(False)
        self.zoom_slider.setEnabled(True)
        if hasattr(self, 'right_controls_container'):
            self.right_controls_container.setEnabled(True)
        self._update_status_bar()
        # 更新监控区显示
        self.video_label.setText("相机已启动\n（等待画面...）")
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #2d2d2d;
                color: #27ae60;
                font-size: 16px;
            }
        """)
    
    def on_camera_closed(self):
        """相机关闭信号响应"""
        if self.ocr_timer is not None:
            self.ocr_timer.stop()
        if self.video_timer is not None:
            self.video_timer.stop()
            self.video_timer = None
        self.switch_camera.blockSignals(True)
        self.switch_camera.setValue(0)
        self.switch_camera.blockSignals(False)
        self.zoom_slider.setEnabled(False)
        if hasattr(self, 'right_controls_container'):
            self.right_controls_container.setEnabled(False)
        self._update_status_bar()
        # 退出提示模式
        self.hint_mode = None
        self._hint_rect_pressing = False
        self.hint_rect_start = None
        self.hint_rect_end = None
        self.hint_points = []
        if hasattr(self, 'btn_ocr_click_hint') and self.btn_ocr_click_hint.text() == "确认":
            self.btn_ocr_click_hint.setText("坐标提示")
        self.video_label.setCursor(Qt.ArrowCursor)
        self.video_label.clear_overlay()
        # 恢复监控区显示为默认状态
        self.video_label.clear()
        self.video_label.setText("实时监控区\n（等待相机启动...）")
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #2d2d2d;
                color: #888;
                font-size: 16px;
            }
        """)
        self.video_label.setAlignment(Qt.AlignCenter)
    
    def on_camera_error(self, error_msg):
        """相机错误信号响应"""
        self.video_label.setText(f"相机错误\n{error_msg}")
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #2d2d2d;
                color: #e74c3c;
                font-size: 16px;
            }
        """)
    
    def on_frame_ready(self, frame):
        """视频帧就绪信号响应（可选，当前使用定时器读取）"""
        pass
    
    def on_ocr_regions_updated(self, regions):
        """检测到的文本区域更新：转换为 label 坐标并绘制淡绿色覆盖层"""
        label_polys = []
        for r in regions:
            poly = []
            for fx, fy in r.polygon:
                lx, ly = self._frame_to_label(fx, fy)
                poly.append((lx, ly))
            if len(poly) >= 3:
                label_polys.append(poly)
        self.video_label.set_regions(label_polys)
        # 可选：更新监测结果表格
        items = [(r.id, f"区域{r.id}") for r in regions]
        self.set_ocr_results(items)
    
    def on_ocr_regions_cleared(self):
        """刷新提示后清除覆盖层"""
        self.video_label.clear_overlay()
        self.set_ocr_results([])
    
    def on_ocr_recognition_started(self):
        """OCR识别开始信号响应"""
        if hasattr(self, 'switch_ocr'):
            self.switch_ocr.blockSignals(True)
            self.switch_ocr.setValue(1)
            self.switch_ocr.blockSignals(False)
        self._update_status_bar()
    
    def on_ocr_recognition_stopped(self):
        """OCR识别停止信号响应"""
        if hasattr(self, 'switch_ocr'):
            self.switch_ocr.blockSignals(True)
            self.switch_ocr.setValue(0)
            self.switch_ocr.blockSignals(False)
        self._update_status_bar()
    
    def on_enhancement_started(self):
        """增强启动信号响应"""
        if hasattr(self, 'switch_enhance'):
            self.switch_enhance.blockSignals(True)
            self.switch_enhance.setValue(1)
            self.switch_enhance.blockSignals(False)
        self._update_status_bar()
    
    def on_enhancement_stopped(self):
        """增强停止信号响应"""
        if hasattr(self, 'switch_enhance'):
            self.switch_enhance.blockSignals(True)
            self.switch_enhance.setValue(0)
            self.switch_enhance.blockSignals(False)
        self._update_status_bar()
    
    def on_user_logged_in(self, username):
        """用户登录信号响应"""
        self.user_display.setText(username)
        self.btn_login.setEnabled(False)
        self.btn_logout.setEnabled(True)
    
    def on_user_logged_out(self):
        """用户退出信号响应"""
        self.user_display.setText("未登录")
        self.btn_login.setEnabled(True)
        self.btn_logout.setEnabled(False)
    
    def on_settings_camera_changed(self, camera_id: int):
        """设置中相机变更信号响应"""
        # 如果相机正在运行，需要重新打开新相机
        if self.camera_manager.is_opened():
            self.camera_manager.close_camera()
            # 可以在这里自动打开新相机，或者提示用户手动打开
            print(f"[设置] 相机已变更为 {camera_id}，请手动重新打开相机")
    
    def on_settings_saved(self):
        """设置保存信号响应"""
        config = self.settings_manager.get_config()
        print(f"[设置] 设置已保存，当前相机ID: {config.camera_id}")
        self._load_det_model()
        self._load_rec_model()
