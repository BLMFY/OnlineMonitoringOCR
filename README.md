# 在线监测系统 (OnlineMonitoring)

基于 OCR 的智能监测系统，支持实时视频/图像识别、预警阈值监测与报警记录管理。采用 PyQt5 构建图形界面，集成文本检测、OCR 识别与图像增强算法。

## 技术栈

- Python 3.7+
- PyQt5
- OpenCV
- NumPy

## 主要功能

### 主监控界面

- **实时监控**：视频流或测试素材（图像/视频）显示，支持 16:9 宽高比自适应
- **待测区域选择**：
  - **框选目标**：依次点击 4 个点围成四边形区域
  - **区域提示**：拖拽矩形框，自动检测框内文本区域
  - **坐标提示**：点击若干点，检测包含这些点的文本区域
- **OCR 监测**：周期性识别各待测区域，实时展示识别结果
- **图像增强**：对画面进行增强处理
- **刷新提示**：清除所有待测区域与预警配置

### 日志管理

- **监测动态**：实时展示各目标的识别值、识别文本与状态（正常/超出上限/超出下限）
- **预警设置**：按目标序号配置上下限阈值；支持序号范围校验
- **监控记录**：历史识别记录查询
- **报警记录**：超阈值报警记录查询与导出

### 测试模式

在设置 → 相机设置 → 测试模式 中可启用测试素材，主界面“开启相机”时将显示选定的图像或视频文件，便于离线调试。

## 环境要求

- Python 3.7 及以上
- 摄像头（或启用测试模式使用本地素材）

## 安装

```bash
# 安装依赖
pip install -r requirements.txt
```

依赖版本（参见 `requirements.txt`）：

- PyQt5 >= 5.15.0
- opencv-python >= 4.5.0
- numpy >= 1.19.0

## 运行

```bash
python main.py
```

## 配置说明

配置文件为项目根目录下的 `config.json`，主要包含：

| 配置项 | 说明 |
|--------|------|
| camera_id | 相机设备 ID |
| camera_width / camera_height | 分辨率 |
| test_mode_enabled | 是否启用测试模式 |
| test_material_path | 测试素材路径（图像或视频） |
| text_detection_model / ocr_model | 模型选择 |
| log_path / data_path | 日志与数据存储路径 |

可通过主界面的 **设置** 入口进行修改，保存后生效。

## 模型与权重

系统使用以下模型：

- **文本检测**：权重位于 `method/weight/det/`（如 `det_db_mbv3_new.pth`）
- **OCR 识别**：权重与字典位于 `method/weight/rec/`（如 `chen/chen_crnn_mbv3.pth`、`chen/chen.txt`）

在 **设置 → 模型配置** 中选择对应模型；若权重文件缺失，需将文件放入上述目录并确认路径配置正确。

## 项目结构

```
OnlineMonitoring8.0/
├── main.py              # 程序入口
├── config.json          # 配置文件
├── requirements.txt     # 依赖列表
├── core/                # 核心模块
│   ├── camera_manager.py    # 相机与测试素材管理
│   ├── ocr_processor.py     # OCR 检测与识别
│   ├── log_manager.py       # 日志与报警管理
│   ├── model_manager.py     # 模型路径管理
│   ├── settings_manager.py  # 配置管理
│   └── ...
├── ui/                  # 界面
│   ├── main_window.py   # 主窗口
│   ├── log_window.py    # 日志窗口
│   ├── settings_window.py  # 设置窗口
│   └── widgets.py       # 自定义控件
└── method/              # 算法与模型
    ├── det_infer.py     # 检测推理
    ├── rec_infer.py     # 识别推理
    └── weight/          # 模型权重目录
        ├── det/         # 检测模型
        └── rec/         # 识别模型与字典
```

## 使用流程

1. 启动程序，进入主界面
2. 打开相机，或启用测试模式并选择测试素材
3. 通过 **框选目标** / **区域提示** / **坐标提示** 确定待测文本区域（可多次添加）
4. 在 **日志 → 预警设置** 中为目标序号配置上下限阈值（可选）
5. 开启 **OCR 监测**，在 **日志 → 监测动态** 查看实时识别结果与状态
6. 需要清空所有区域与预警配置时，点击 **刷新提示**

## 技术文档

更详细的界面架构与算法调用流程见 `technical_report.md`。
