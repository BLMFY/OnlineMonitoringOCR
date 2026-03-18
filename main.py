"""
智能检测系统主程序入口
"""
import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle('Fusion')
    
    # 创建并显示主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

