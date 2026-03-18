"""
用户管理模块
负责用户登录、退出等功能
"""
from PyQt5.QtCore import QObject, pyqtSignal


class UserManager(QObject):
    """用户管理类"""
    
    # 信号定义
    user_logged_in = pyqtSignal(str)  # 用户登录信号，参数为用户名
    user_logged_out = pyqtSignal()  # 用户退出信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_user = None  # 当前用户
        self.is_logged_in = False  # 登录状态标志
        
    def login(self, username, password=None):
        """
        用户登录（占位实现）
        
        Args:
            username: 用户名
            password: 密码（可选）
            
        Returns:
            bool: 登录成功返回True，失败返回False
        """
        print(f"[用户管理] 用户登录: {username}")
        # TODO: 实现登录逻辑（验证用户名密码等）
        
        # 占位实现：直接登录成功
        self.current_user = username
        self.is_logged_in = True
        self.user_logged_in.emit(username)
        return True
        
    def logout(self):
        """用户退出"""
        print("[用户管理] 用户退出")
        username = self.current_user
        self.current_user = None
        self.is_logged_in = False
        self.user_logged_out.emit()
        return username
        
    def get_current_user(self):
        """
        获取当前用户
        
        Returns:
            str: 当前用户名，未登录返回None
        """
        return self.current_user if self.is_logged_in else None
        
    def is_user_logged_in(self):
        """
        检查是否已登录
        
        Returns:
            bool: 已登录返回True，否则返回False
        """
        return self.is_logged_in

