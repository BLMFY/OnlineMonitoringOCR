import json
import re
import wmi
import pandas as pd
import datetime
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2
import os
import sys
import time
import threading
import pathlib
import numpy as np
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))
from det_infer import DetInfer
from rec_infer import RecInfer, number_ocr, point_sort, get_rotate_crop_image, Add_text
from window1 import Ui_MainWindow , Ui_Dialog1, Ui_Dialog_ex, Ui_Dialog_op, Ui_Dialog_sh, Ui_Dialog_false
from ov_ocr import det_ov, rec_ov, to_detbox
from tcp import sendTCP, teamsendTCP, exchange, receive_tcp, Ping
from wave_1 import wave_detect
from compare import compare
from func_timeout import func_set_timeout
import socket

th_max = []
th_min = []
th_name = []
th_d = []

command_close_all=["01","05","00","00","00","00","CD","CA"]#全关
command_green = ["01","05","00","03","FF","00","7C","3A"]#绿灯亮
command_yellow=["01","05","00","1B","FF","00","FC","3D"]#黄灯亮+绿灯亮
command_red=["01","05","00","1A","FF","00","AD","FD"]#红灯亮
from ctrl_light import send_light

# 画虚线
def dot_line(img,pt1,pt2,color,thickness=1,dot=1 ,gap=10): 
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5 
    pts= [] 
    for i in np.arange(0,dist,gap): 
        r=i/dist 
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5) 
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5) 
        p = (x,y) 
        pts.append(p) 
 
    if dot: 
        for p in pts: 
            cv2.circle(img,p,thickness,color,-1) 
    else: 
        s=pts[0] 
        e=pts[0] 
        i=0 
        for p in pts: 
            s=e 
            e=p 
            if i%2==1: 
                cv2.line(img,s,e,color,thickness) 
            i+=1

# 画虚线框
def dot_rect(img, p1, p2, color, thickness):
    x0, y0 = p1[0], p1[1]
    x1, y1 = p2[0], p2[1]
    dot_line(img, (x0, y0), (x1, y0), color, thickness, 0, gap=10)
    dot_line(img, (x1, y0), (x1, y1), color, thickness, 0, gap=10)
    dot_line(img, (x1, y1), (x0, y1), color, thickness, 0, gap=10)
    dot_line(img, (x0, y1), (x0, y0), color, thickness, 0, gap=10)

# 判断点是否在四边形内 
def isPointInRect(x, y, rect):
    a = (rect[2] - rect[0])*(y - rect[1]) - (rect[3] - rect[1])*(x - rect[0])
    b = (rect[4] - rect[2])*(y - rect[3]) - (rect[5] - rect[3])*(x - rect[2])
    c = (rect[6] - rect[4])*(y - rect[5]) - (rect[7] - rect[5])*(x - rect[4])
    d = (rect[0] - rect[6])*(y - rect[7]) - (rect[1] - rect[7])*(x - rect[6])
    if ((a>0)&(b>0)&(c>0)&(d>0))or((a<0)&(b<0)&(c<0)&(d<0)):
        return True
    else:
        return False

# 判断点是否在矩形内
def isPointInBox(x, y, box):
    if (x >= box[0])&(y >= box[1])&(x <= box[2])&(y <= box[3]):
        return 1
    else:
        return 0

# 判断线段是否在矩形内
def isLineInBox(line, box):
    if isPointInBox(line[0],line[1],box) and isPointInBox(line[2],line[3],box):
        return 1
    else:
        return 0
    
# 检查波形检测划线是否符合规则
def check_line(boxes, lines):
    if len(boxes) == len(lines):
        # new_lines = copy.deepcopy(lines)
        new_lines = []
        for b in boxes:
            for l in lines:
                if isLineInBox(l,b):
                    new_lines.append(l)
                    break
        return new_lines
    else:
        return lines
    
# 以下代码为调用相机镜头
def gstreamer_pipeline(capture_width=1280, capture_height=960, display_width=1280, display_height=960, framerate=20, flip_method=0, sensor_id=0):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (sensor_id, capture_width, capture_height, framerate, flip_method, display_width, display_height,)
    )

# 标定图像
def undistort(frame):

    k = np.array([[1339.11779, 0., 679.314522],
                  [0., 1786.64230, 478.474468],
                  [0., 0., 1.0]])

    d = np.array([-0.51314196, 0.53424932,
                 0.00187583, 0.00216236, -0.82117518])
    h, w = frame.shape[:2]
    mapx, mapy = cv2.initUndistortRectifyMap(k, d, None, k, (w, h), 5)
    return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

# 数据比较
def compare_list_elements(list_1, list_Max, list_Min):  #判断list中相同位置数字大小
    list_1 = [re.sub("[^0-9,.]", "0", s) for s in list_1]
    #list_1= [ int(x) for x in list_1 ]:
    safe = True
    alarm_list = []
    try:
        list_1= [ float(x) for x in list_1 ]
        # print(list_1)
        for i, (l1, lmx ,lmn) in enumerate(zip(list_1, list_Max, list_Min)):
            if l1 == '/' or l1 == '-' or lmx == '-' or lmn == '-':
                continue
            if (l1 > lmx) or (l1 < lmn):
                alarm_list.append(i)
                safe = False
        return safe , alarm_list
    except:
        print("字符类型报错！")
        return safe , alarm_list

@func_set_timeout(3)
def read_url(url):
    capture = cv2.VideoCapture(url)
    return capture

class Stats(QWidget, Ui_MainWindow):
    
    def __init__(self):
        # 从文件中加载UI定义

        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.button , self.textEdit
        super().__init__()
        self.setupUi(self)
        self.setWindowIcon(QIcon('OnlineMonitoring/yigeai.png'))
        # self.det_model=[
        #         {
        #         'title':'正弦波',
        #         'pt':'pt_dir/yolov5s.pt',
        #         'cfg':'data/coco.yaml'
        #         } ,
        #         {
        #         'title':'矩形波',
        #         'pt':'pt_dir/jingujian.pt',
        #         'cfg':'data/jingujian.yaml'
        #         } ]
        # self.rec_model=[
        #         {
        #         'title':'中英文',
        #         'model_path':'OnlineMonitoring/pt_dir/ch_rec_mobile_crnn_mbv3.pth',
        #         'trt_path':{'backbone':'OnlineMonitoring/rec_dir/rec_b.pth',
        #                     'head':'OnlineMonitoring/rec_dir/rec_h.pth'},
        #         'dict':'OnlineMonitoring/pt_dir/ppocr_keys_v1.txt'
        #         },
        #         {
        #         'title':'数码管',
        #         'model_path':'OnlineMonitoring/pt_dir/num_tube_rec.pth',
        #         'trt_path':{'backbone':'OnlineMonitoring/rec_dir/rec_b.pth',
        #                     'head':'OnlineMonitoring/rec_dir/rec_h.pth'},
        #         'dict':'OnlineMonitoring/pt_dir/ppocr_keys_v1.txt'
        #         } ]
        self.det_model=[
                {
                'title':'正弦波',
                'pt':'pt_dir/yolov5s.pt',
                'cfg':'data/coco.yaml'
                } ,
                {
                'title':'矩形波',
                'pt':'pt_dir/jingujian.pt',
                'cfg':'data/jingujian.yaml'
                } ]
        self.rec_model=[
                {
                'title':'中英文',
                'model_path':'pt_dir/ch_rec_mobile_crnn_mbv3.pth',
                'trt_path':{'backbone':'rec_dir/rec_b.pth',
                            'head':'rec_dir/rec_h.pth'},
                'dict':'pt_dir/ppocr_keys_v1.txt'
                },
                {
                'title':'数码管',
                'model_path':'pt_dir/num_tube_rec.pth',
                'trt_path':{'backbone':'rec_dir/rec_b.pth',
                            'head':'rec_dir/rec_h.pth'},
                'dict':'pt_dir/ppocr_keys_v1.txt'
                } ]
        # det
        self.det_num = 0
        self.det_arg = {
            'conf':0.60,
            'iou':0.60,
            'model':self.det_model[self.det_num]          
        }

        # rec
        self.rec_num = 1

        self.rec_arg = {
            'conf':0.80,
            'model':self.rec_model[self.rec_num]
        }

        self.rec_arg0 = {
            'conf':0.80,
            'model':self.rec_model[0]
        }

        self.rec_arg1 = {
            'conf':0.80,
            'model':self.rec_model[1]
        }

        self.initial_win()

        self.rec0 = RecInfer(model_path=self.rec_arg0['model']['model_path'],
                                dict_path=self.rec_arg0['model']['dict'])
        self.rec1 = RecInfer(model_path=self.rec_arg1['model']['model_path'],
                                dict_path=self.rec_arg1['model']['dict'])

        self.word_sear = DetInfer(model_path='pt_dir/det_db_mbv3_new.pth')
        self.cap = None
        self.video = None

        # 获取时间
        self.timedisplay = ''
        self.timer = QTimer()
        self.timer.timeout.connect(self.showtime)
        # 摄像头
        self.win_streams = ['rtsp://admin:34567@192.168.1.10:554/mpeg4/ch1/main/av_stream',
                            'rtsp://admin:34567@192.168.1.11:554/mpeg4/ch1/main/av_stream',
                            'rtsp://admin:34567@192.168.1.12:554/mpeg4/ch1/main/av_stream',
                            'rtsp://admin:34567@192.168.1.13:554/mpeg4/ch1/main/av_stream',
                            0]
        self.win_stream = self.win_streams[0]
        self.recording = 0
        self.saving = 0
        self.save_result = None
        self.save_time = None
        self.text_rec = 0
        self.mate_rec = 0
        self.pic_compare = 0
        self.wave_mode = 0
        self.box_mode = 'text'
        self.video_path = 'C:\\Users\\Administrator\\Desktop\\Video'
        self.data_path = 'C:\\Users\\Administrator\\Desktop\\Data'
        self.pic_path = 'C:\\Users\\Administrator\\Desktop\\Image'
        # self.pic_path = 'F:\OnlineMonitoring6.3-pink\OnlineMonitoring\Image'
        self.sq = []

        #通道筛选
        self.td1_yz_1 = 160  #通道1阈值初始为160
        self.td1_ys_1 = 2  #通道1颜色分类初始为2
        self.td2_yz_1 = 160
        self.td2_ys_1 = 2
        self.td3_yz_1 = 160
        self.td3_ys_1 = 2
        self.td4_yz_1 = 160
        self.td4_ys_1 = 2
        self.td1_yz.setText("160")
        self.td1_ys.setText("2")
        self.td2_yz.setText("160")
        self.td2_ys.setText("2")
        self.td3_yz.setText("160")
        self.td3_ys.setText("2")
        self.td4_yz.setText("160")
        self.td4_ys.setText("2")

        #波形占空比检测参数初始化
        self.RGB = [2, 2, 2, 2]
        self.colorThresh = [160, 160, 160, 160]
        self.jcyz.setText("160")
        self.jcys.setText("2")
        self.wave_disp_tab = [self.td_display_1, self.td_display_2, self.td_display_3, self.td_display_4]
        self.wave_txt_tab = [self.td_tab_1, self.td_tab_2, self.td_tab_3, self.td_tab_4]

        #报警延时时间
        self.th = 2      #默认延时为2s
        self.alarm_time_pv.setText('{} s'.format(self.th))

        #像素对比
        self.limit_value = 20  #初始像素限值为50；
        self.limit_value_now.setText(str(self.limit_value))
        self.err_rate = "00.0"

        # 远程控制
        self.control_recording = 0
        self.control_recording_change = 0
        self.control_saving_change = 0
        self.control_change = False
        self.control_count = 0
        self.control_recorder = 0
        self.alarm_stop_change = True
        self.web_connect = False

        self.prop_rec_box = [] # 提示框体内的检测框
        self.over_record = Dialog1()
        self.win_false = Dialog_false()
        self.win_op = Dialog_op()
        self.win_sh = Dialog_sh()
        self.win_ex = Dialog_ex()
        self.stopEvent = threading.Event()
        self.stopEvent.clear()

    # 按钮绑定对应函数
        self.open_cam.clicked.connect(self.open_camera)
        self.shut_cam.clicked.connect(lambda:self.shut_camera(True))       
        self.lock_all.clicked.connect(self.all_lock) 
        self.startLine.clicked.connect(self.draw_line)
        self.Boxing_start.clicked.connect(self.det_start)
        self.select_box.clicked.connect(self.draw_box)
        self.select_area.clicked.connect(self.point_box)
        self.startButton2.clicked.connect(self.rec_start)
        self.stopButton2.clicked.connect(self.rec_stop)
        self.proposal_box.clicked.connect(self.box_prompt)
        self.proposal_point.clicked.connect(self.point_prompt)
        self.clean_box.clicked.connect(self.clean_all)
        self.search_box.clicked.connect(self.auto_search)
        self.video_start.clicked.connect(self.start_video)
        self.video_query.clicked.connect(self.open_file)
        self.th_Button.clicked.connect(self.Add_Th_list)
        self.save_start.clicked.connect(self.save_open)
        self.alarm_start.clicked.connect(self.open_alarm)
        self.set_para.clicked.connect(self.set_parameter) # 设置波形检测的参数
        self.remote.clicked.connect(self.remote_control) # 开启远程通讯功能
        self.ocr_dis.clicked.connect(self.ocr_flag_xg) # 是否在界面显示OCR实时数据
        self.change_path_1.clicked.connect(lambda: self.path_change('v'))
        self.change_path_2.clicked.connect(lambda: self.path_change('d'))
        self.change_path_3.clicked.connect(lambda: self.path_change('i'))

        self.over_record.ok.clicked.connect(self.control_recorder_) # 更改服务器接受主题
        self.over_record.check.clicked.connect(self.open_file)
        self.alarm_time_pushButton.clicked.connect(self.alarm_time) # 设置报警延时时间

        self.set_para_td1.clicked.connect(self.set_para_1) # 通道1参数按钮
        self.set_para_td2.clicked.connect(self.set_para_2)
        self.set_para_td3.clicked.connect(self.set_para_3)
        self.set_para_td4.clicked.connect(self.set_para_4)

        #像素对比
        self.select_target_Button.clicked.connect(self.select_img) # 选中目标图像
        self.start_compare_Button.clicked.connect(self.start_compare) # 开始对比
        self.stop_compare_Button.clicked.connect(self.stop_compare) # 停止对比
        self.start_save_Button.clicked.connect(self.strat_save) # 开始保存
        self.stop_save_Button.clicked.connect(self.stop_save) # 停止保存
        self.limit_value_set_Button.clicked.connect(self.set_limit_value) # 设置限制值

        # self.num_tube.clicked.connect(lambda: self.change_model(1))
        # self.eng_ch.clicked.connect(lambda: self.change_model(0))
        self.text_b.clicked.connect(lambda: self.change_box('text'))
        self.wave_b.clicked.connect(lambda: self.change_box('wave'))
        self.compare_b.clicked.connect(lambda: self.change_box('compare'))
        self.win_choice.currentIndexChanged.connect(self.channel_choice)


        # 退出信号
        self.quit.clicked.connect(self.quit_out)

    # 初始化设置
    def initial_win(self):
        # self.cap = None
        self.open_cam.setEnabled(True)
        self.shut_cam.setEnabled(False)
        self.startLine.setEnabled(False)#用来做波形检测
        self.Boxing_start.setEnabled(False)
        self.startButton2.setEnabled(False)
        self.stopButton2.setEnabled(False)
        self.det_Box.setEnabled(0)
        self.rec_Box.setEnabled(0)
        self.clean_box.setEnabled(False)
        #self.rec_select.setChecked(True)
        #self.mode_slt.setEnabled(False)
        self.video_query.setEnabled(True)
        self.video_start.setEnabled(False)
        self.save_start.setEnabled(False)
        self.alarm_start.setEnabled(False)
        self.th_Button.setEnabled(False)
        self.not_stop = True
        self.alarm_stop = True#控制报警开启
        self.alarm_save = False#警报保存标志
        self.color = send_light(command_green)#初始化报警灯
        if self.color:
            self.color = "green"
        else:
            mesBox = QMessageBox()
            mesBox.setWindowTitle('提示')
            mesBox.setText('报警灯未连接！')
            mesBox.setIcon(QMessageBox.Information)
            mesBox.setStandardButtons(QMessageBox.Yes)
            mesBox.setStyleSheet("QPushButton:hover{background-color: rgb(255, 93, 52);}")
            mesBox.exec_()
            # QMessageBox.information(self, '提示', '报警灯未连接!')
        #self.led_cs.setEnabled(0)
        self.flag_search = False
        self.txt_tab.lower()
        self.txt_tab.clear()
        self.wave_tab.lower()
        self.wave_tab.clear()
        self.DisplayLabel.flag_paint = False
        self.DisplayLabel.key = 0
        self.DisplayLabel.edit_box = None
        self.DisplayLabel.edit_id = 0
        self.DisplayLabel.rec_box = []
        self.DisplayLabel.rec_area = []
        self.DisplayLabel.setCursor(Qt.ArrowCursor)
        self.data_dis.setRowCount(1)
        self.data_dis.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)#铺满
        self.data_dis.setEditTriggers(QAbstractItemView.NoEditTriggers)#禁止编辑

        #以下为初始化报警记录显示窗口
        self.alarm_list.setRowCount(0)
        self.alarm_list.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)#铺满
        self.alarm_list.setEditTriggers(QAbstractItemView.NoEditTriggers)#禁止编辑
        self.alarm_data = []  #用于数据传递报警位置
        self.alarm_flag = [0]*100  #用于记录上次报警标识

        #远程app参数初始化
        # IP 和端口
        self.server_ip = 'bemfa.com'
        self.server_port = 8344

        #接收报警关闭信号
        self.alarm = True
        self.remote_flag = False#远程功能的标志位
        self.ocr_output_flag = True  #界面显示识别数据的标志位
        send_light(command_close_all)

        #像素对比
        self.select_img_flag = False
        self.compare_flag  = False       #比较标志位
        self.compare_save_flag = False   #图像保存标志位

    # 时间显示
    def showtime(self):
        time = QDateTime.currentDateTime()  # 获取当前时间
        self.timedisplay = time.toString("yyyy-MM-dd hh:mm:ss dddd")  # 格式化一下时间
        timedisplay = self.timedisplay + ' '
        self.time.setText(timedisplay)

    def start_timer(self):
        self.timer.start(1000)  # 每隔一秒刷新一次，这里设置为1000ms

    def channel_choice(self, index):
        self.win_stream = self.win_streams[index]

    # 打开相机    
    def open_camera(self):
        

        try:  
            # 调用读取URL的函数
            self.cap = read_url(self.win_stream)
        except:
            self.cap = None
            self.win_false.show()
            QTimer.singleShot(1500, self.win_false.close)
            print('超时！')

        if self.cap:
            self.cap = cv2.VideoCapture(self.win_stream)
            self.open_cam.setEnabled(False)
            self.shut_cam.setEnabled(True)
            self.video_start.setEnabled(True)
            # self.startLine.setEnabled(True)
            self.Boxing_start.setEnabled(True)
            # self.video_query.setEnabled(False)
            self.rec_mode(initial=True)
            self.det_Box.setEnabled(1)
            self.win_op.show()
            self.clean_box.setEnabled(True)
            self.set_button([1,1,1,1])
    
            QTimer.singleShot(1500, self.win_op.close)
            # self.cap = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0), cv2.CAP_GSTREAMER)  # 调用相机摄像头
            # self.cap = cv2.VideoCapture('rtsp://admin:34567@192.168.1.12:554/mpeg4/ch1/main/av_stream')
            # self.cap = cv2.VideoCapture('rtsp://admin:12345@192.168.1.68:554/ Stream/Live/101')
            th = threading.Thread(target=self.Display)
            th.start()


    # 相机关闭       
    def shut_camera(self, win=False):
        self.alarm = False
        #关闭远程标志位，停止接收数据标志位复位，停止接收
        self.remote_flag = False
        self.rev_stop = False
        # self.alarm_stop = exchange(self.alarm_stop,self.rev_stop)#接收的远程报警开关状态
        self.rec_stop()

        if win:
            self.win_sh.show()
            self.text_rec = 0
            self.stopEvent.set()
            QTimer.singleShot(1500, self.win_sh.close)
            self.initial_win()
        else:
            self.text_rec = 0
            # 关闭事件设为触发，关闭视频播放
            self.stopEvent.set()
            time.sleep(1)
            self.initial_win()

    # 录制功能
    def start_video(self):
        if self.video_start.text() == '开始录制':
            self.control_recording = 1 
            self.recording = 1
            # fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # print(self.video_path  + self.timedisplay + '.mp4')

            self.save_result=cv2.VideoWriter(self.video_path + '\\' + self.timedisplay.replace(':','_') + '.mp4', fourcc, 20.0, (1920, 1080))
            self.video_start.setText('结束录制')
        elif self.video_start.text() == '结束录制':
            # self.recording = 0
            self.over_record.show()
            self.control_recording = 0
            self.recording = 2
            self.save_result.release()
            # self.save_result = None
            self.video_start.setText('开始录制')

    # 按钮锁定
    def all_lock(self):
        if self.lock_all.text() == 'LOCK':
            self.login.setEnabled(0)
            self.quit.setEnabled(0)
            self.operate.setEnabled(0)
            self.lock_all.setText("UNLOCK")
        
        elif self.lock_all.text() == 'UNLOCK':
            self.login.setEnabled(1)
            self.quit.setEnabled(1)
            self.operate.setEnabled(1)
            self.lock_all.setText("LOCK")

    # 查看录制
    def open_file(self):
        # source = QFileDialog.getOpenFileName(self, '选取视频或图片', os.getcwd(), "Pic File(*.mp4 *.mkv *.avi *.flv "
        #                                                                    "*.jpg *.png)")
        if self.video_query.text() == '查看录制':
            self.frame_rate = QTimer()
            # config_file = 'OnlineMonitoring/demo/video.json'
            # config = json.load(open(config_file, 'r', encoding='utf-8'))
            # open_fold = config['open_fold']
            open_fold = self.video_path
            if not os.path.exists(open_fold):
                open_fold = os.getcwd()
            self.fileName, _ = QFileDialog.getOpenFileName(self, '选取视频或图片', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                           "*.jpg *.png)")
            if self.fileName:
                if self.cap:
                    self.shut_camera()
                self.not_stop = True
                self.video_query.setText('退出观看')
                #self.det_Box.setEnabled(0)
                self.rec_Box.setEnabled(0)
                self.open_cam.setEnabled(0)
                self.video_start.setEnabled(0)
                source = self.fileName
                self.video = cv2.VideoCapture(source)
                self.frameRate = self.video.get(cv2.CAP_PROP_FPS)
                self.frame_rate.start(int(1000 / self.frameRate))
                self.frame_rate.timeout.connect(self.Video_display)
        elif self.video_query.text() == '退出观看':
            self.frame_rate.stop()
            self.not_stop = False
            self.video = None
            self.video_query.setText('查看录制')
            self.DisplayLabel.clear()
            self.DisplayLabel.setText("实时监视区")
            self.initial_win()

    # 录像播放
    def Video_display(self):
        ret, image = self.video.read()
        if ret and self.not_stop:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                vedio_img = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
            elif len(image.shape) == 1:
                vedio_img = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_Indexed8)
            else:
                vedio_img = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
 
            self.DisplayLabel.setPixmap(QPixmap(vedio_img))
            self.DisplayLabel.setScaledContents(True)  # 自适应窗口
        else:
            self.video.release()
            self.frame_rate.stop()
            self.video_query.setText('查看录制')
            self.DisplayLabel.clear()
            self.DisplayLabel.setText("实时监视区")
            self.text_rec = 0
            self.initial_win()
   
    # 按钮计划
    def set_button(self, 
                   order: list): 
        self.select_box.setEnabled(order[0])
        self.select_area.setEnabled(order[1])
        self.proposal_box.setEnabled(order[2])
        self.proposal_point.setEnabled(order[3])

    # 更换框选模式
    def change_box(self, item):
        self.DisplayLabel.box_mode = item
        if item == 'text':
            self.select_area.setEnabled(1)
        else:
            self.select_area.setEnabled(0)

    # 识别模式画框
    def draw_box(self):
        if self.select_box.text() == '框选区域':
            self.DisplayLabel.key = 1
            self.DisplayLabel.setCursor(Qt.CrossCursor)
            self.select_box.setText('取消')

        elif self.select_box.text() == '删除区域':
            if self.DisplayLabel.edit_id <= len(self.DisplayLabel.rec_box):
                self.DisplayLabel.rec_box.remove(self.DisplayLabel.edit_box)
            elif self.DisplayLabel.edit_id <= len(self.DisplayLabel.rec_box)+len(self.DisplayLabel.wave_box):
                self.DisplayLabel.wave_box.remove(self.DisplayLabel.edit_box)
                if self.DisplayLabel.edit_line:
                    self.DisplayLabel.line.remove(self.DisplayLabel.edit_line)
                self.DisplayLabel.line = check_line(self.DisplayLabel.wave_box, self.DisplayLabel.line)
            else:
                self.DisplayLabel.compare_box = []
            self.DisplayLabel.edit_box = None
            self.select_box.setText('框选区域')
            self.startButton2.setEnabled(True)
            self.DisplayLabel.key = 2
            self.set_button([1,1,1,1])

        elif self.select_box.text() == '取消':
            self.DisplayLabel.key = 2
            self.DisplayLabel.setCursor(Qt.ArrowCursor)
            self.select_box.setText('框选区域')
            self.set_button([1,1,1,1])

    # 点选四边形       
    def point_box(self):
        if self.select_area.text() == '四点区域':
            self.DisplayLabel.key = 5
            self.DisplayLabel.setCursor(Qt.CrossCursor)
            self.select_area.setText('取消')

        elif self.select_area.text() == '删除区域':
            if self.DisplayLabel.edit_id <= len(self.DisplayLabel.rec_box):
                self.DisplayLabel.rec_box.remove(self.DisplayLabel.edit_box)
            elif self.DisplayLabel.edit_id <= len(self.DisplayLabel.rec_box)+len(self.DisplayLabel.wave_box):
                self.DisplayLabel.wave_box.remove(self.DisplayLabel.edit_box)
            else:
                self.DisplayLabel.compare_box = []
            self.DisplayLabel.edit_box = None
            self.select_area.setText('四点区域')
            self.startButton2.setEnabled(True)
            self.set_button([1,1,1,1])

        elif self.select_area.text() == '取消':
            self.DisplayLabel.key = 2
            self.DisplayLabel.rec_area = []
            self.DisplayLabel.x0,self.DisplayLabel.x1,self.DisplayLabel.y0,self.DisplayLabel.y1 = 0,0,0,0
            self.DisplayLabel.setCursor(Qt.ArrowCursor)
            self.select_area.setText('四点区域')
            self.set_button([1,1,1,1])

    # 框提示区域
    def box_prompt(self):
        if self.proposal_box.text() == '区域建议':
            self.DisplayLabel.key = 8
            self.DisplayLabel.setCursor(Qt.CrossCursor)
            self.proposal_box.setText('取消')
            self.set_button([0,0,1,0])
            self.search_box.setEnabled(0)

        elif self.proposal_box.text() == '取消' or self.proposal_box.text() == '确认':
            self.DisplayLabel.key = 2
            if self.proposal_box.text() == '确认':
                for box in self.prop_rec_box:
                    self.DisplayLabel.rec_box.append(box)
                self.prop_rec_box = []
                self.proposal_point.setText('点击建议')
                self.DisplayLabel.key = 4
            self.DisplayLabel.prop_box = []
            self.DisplayLabel.x0,self.DisplayLabel.x1,self.DisplayLabel.y0,self.DisplayLabel.y1 = 0,0,0,0
            self.DisplayLabel.setCursor(Qt.ArrowCursor)
            self.proposal_box.setText('区域建议')
            self.set_button([1,1,1,1])

    # 点提示区域
    def point_prompt(self):
        if self.proposal_point.text() == '点击建议':
            self.DisplayLabel.key = 10
            self.proposal_point.setText('取消')
            self.DisplayLabel.setCursor(Qt.CrossCursor)
            self.set_button([0,0,0,1])
            self.search_box.setEnabled(0)

        elif self.proposal_point.text() == '取消':
            self.DisplayLabel.key = 4
            self.proposal_point.setText('点击建议')
            self.DisplayLabel.test_point = []
            self.DisplayLabel.setCursor(Qt.ArrowCursor)
            self.set_button([1,1,1,1])

        elif self.proposal_point.text() == '撤销':
            self.DisplayLabel.key = 4
            self.DisplayLabel.prop_box = []
            self.prop_rec_box = []
            self.DisplayLabel.x0,self.DisplayLabel.x1,self.DisplayLabel.y0,self.DisplayLabel.y1 = 0,0,0,0
            self.DisplayLabel.setCursor(Qt.ArrowCursor)
            self.proposal_box.setText('区域建议')
            self.proposal_point.setText('点击建议')
            self.select_box.setEnabled(1)
            self.select_area.setEnabled(1)
            self.set_button([1,1,1,1])

    # 清除框选
    def clean_all(self):
        self.DisplayLabel.rec_box = []
        self.DisplayLabel.wave_box = []
        self.DisplayLabel.compare_box = []
        self.DisplayLabel.edit_box = None
        self.select_box.setText('框选区域')
        self.select_area.setText('四点区域')
        self.DisplayLabel.line = []
    
    # 自动全局搜索
    def auto_search(self):
        self.flag_search = True
        self.DisplayLabel.key = 4

    # 识别模型更换
    def change_model(self, item):
        self.rec_num = item
        self.rec_arg['model'] = self.rec_model[self.rec_num]

    # 字符识别开始
    def rec_start(self):
        self.text_rec = 1
        self.mate_rec = 0
        # self.select_box.setEnabled(False)
        
        self.startButton2.setEnabled(False)
        self.stopButton2.setEnabled(True)
        self.save_start.setEnabled(True)
        self.alarm_start.setEnabled(True)
        self.th_Button.setEnabled(True)
        self.txt_tab.raise_()
        # self.rec = RecInfer(model_path=self.rec_arg['model']['model_path'],
        #                     dict_path=self.rec_arg['model']['dict'])

    # 字符识别结束
    def rec_stop(self):
        self.text_rec = 0
        self.mate_rec = 0
        self.select_box.setEnabled(True)
        self.select_area.setEnabled(True)
        self.startButton2.setEnabled(True)
        self.stopButton2.setEnabled(False)
        # self.th_Button.setEnabled(False)
        self.data_dis.clearContents()#识别结束清空实时数据显示区
        #self.mode_slt.setEnabled(True)
        self.DisplayLabel.edit_box = None
        # self.DisplayLabel.rec_box = []
        self.DisplayLabel.key = 2
        self.DisplayLabel.setCursor(Qt.ArrowCursor)
        self.txt_tab.lower()
        self.txt_tab.clear()
        # self.rec = None
        global th_name,th_d,th_max,th_min
        th_d = []
        th_max = []
        th_min = []
        th_name = []
        # self.v_lable.setText("正常")
        # self.led_label.setStyleSheet("background-color: rgb(0, 255, 0);\n"
        #                                                 "border-radius: 17px;\n"
        #                                                 "border: 10px;")

    # 字符识别功能块开启
    def rec_mode(self, initial=False):
        self.rec_Box.setEnabled(1)
        #self.det_Box.setEnabled(0)
        if initial:
            self.stopButton2.setEnabled(0)
    
    # 字符识别结果显示
    def ocr_output(self, txt):
        ocr_disp = ''
        if txt:
            for i in range(len(txt)):
                if self.alarm_data and (i in self.alarm_data):
                    ocr_disp = ocr_disp + ' ' + str(i+1) + '、' + txt[i] + '\n' 
                else:
                    ocr_disp = ocr_disp + ' ' + str(i+1) + '、' + txt[i] + '\n'
            self.txt_tab.setText(ocr_disp)
    
    # 波形检测结果显示
    def zhi_output(self, zhan, num_ocr):
        wave_disp = ''
        for i in range(len(zhan)):
            wave_disp = wave_disp + str(i+1+num_ocr) + '、' + zhan[i] +'\n'
        self.wave_tab.setText(wave_disp)

    def select_result(self, sqq, ocr):
        out0 = self.rec0.predict(ocr)
        out1 = self.rec1.predict(ocr)
        # print('out',out0)
        # print('sq',sqq)
        if len(sqq) == len(out1):
            for i in range(len(out0)):
                if sqq[i]:
                    out0[i] = out1[i] 
        return out0

    # 匹配识别模型
    def sequence_result(self, ocr):
        # print(out0, out1)
        out_sq = []
        out0 = self.rec0.predict(ocr)
        out1 = self.rec1.predict(ocr)
        for i in range(len(out0)):
            if len(out0[i][0][1]) and len(out1[i][0][1]):
                avg0 = sum(out0[i][0][1])/len(out0[i][0][1])
                avg1 = sum(out1[i][0][1])/len(out1[i][0][1])
                if avg1 > max(0.85, avg0):
                    out_sq.append(1)

                else:
                    out_sq.append(0)
        return out_sq

    # 波形画两点线
    def draw_line(self):
        if self.startLine.text() == '开始画线':
            self.DisplayLabel.key = 7
            self.DisplayLabel.setCursor(Qt.CrossCursor)
            self.startLine.setText('取消')
        elif self.startLine.text() == '删除画线':
            self.DisplayLabel.line.remove(self.DisplayLabel.edit_line)
            self.DisplayLabel.line = check_line(self.DisplayLabel.wave_box, self.DisplayLabel.line)
            self.DisplayLabel.edit_line = None
            self.startLine.setText('开始画线')
            self.startButton2.setEnabled(True)
        elif self.startLine.text() == '取消':
            self.DisplayLabel.key = 2
            self.DisplayLabel.rec_line = []
            self.DisplayLabel.x0,self.DisplayLabel.x1,self.DisplayLabel.y0,self.DisplayLabel.y1 = 0,0,0,0
            self.DisplayLabel.setCursor(Qt.ArrowCursor)
            self.startLine.setText('开始画线')

    # 波形检测
    def det_start(self):
        if self.Boxing_start.text() == '开始检测':
            self.wave_mode = 1
            self.wave_tab.raise_()
            self.startLine.setEnabled(0)
            self.DisplayLabel.key = 2
            self.Boxing_start.setText('结束检测')
            self.th_Button.setEnabled(True)

        elif self.Boxing_start.text() == "结束检测":
            self.wave_mode = 0
            self.wave_tab.lower()
            self.wave_tab.clear()
            for td in self.wave_disp_tab:
                td.clear()
            self.td_display_1.setText('通道一')
            self.td_display_2.setText('通道二')
            self.td_display_3.setText('通道三')
            self.td_display_4.setText('通道四')
            self.startLine.setEnabled(1)
            self.DisplayLabel.key = 2
            self.Boxing_start.setText('开始检测')

    # 波形检测参数设置
    def set_parameter(self):                                                         
        self.colorThresh[0] = self.ysyz.value()
        self.RGB[0] = self.ysxz.value()
        self.jcyz.setText('{}'.format(self.colorThresh[0]))
        self.jcys.setText('{}'.format(self.RGB[0]))
        self.td1_yz.setText('{}'.format(self.colorThresh[0]))
        self.td1_ys.setText('{}'.format(self.RGB[0]))

    # 图像像素级比较
    def select_img(self):
        if  self.select_target_Button.text() == '选择目标':
            self.select_target_Button.setText('取消选择')

            s2 = self.target_dis.size()
            self.target_img = self.compare_img
            img2 = cv2.resize(self.target_img, (int(s2.width()//4*4), int(s2.height()//4*4)))
            img2 = QImage(img2.data, img2.shape[1], img2.shape[0], QImage.Format_RGB888)
            self.target_dis.setPixmap(QPixmap.fromImage(img2))
            self.target_dis.show()

        else:
            self.stop_compare()
            self.select_target_Button.setText('选择目标')
            self.target_img = []
            self.target_dis.clear()
            self.target_dis.setText('目标图像')

    # 图像校对开启
    def start_compare(self):  
        self.pic_compare = 1
        self.th_Button.setEnabled(True)
        self.compare_state_dis.setText('校对中')
    
    # 图像校对关闭
    def stop_compare(self):  
        self.pic_compare = 0 
        self.compare_state_dis.setText('停止')
        self.real_dis.clear()
        self.real_dis.setText('实时图像')
    
    # 保存差异图像
    def strat_save(self):
        self.compare_save_flag = True
        self.compare_save_dis.setText('保存中')

    # 关闭差异图像
    def stop_save(self):
        self.compare_save_flag = False
        self.compare_save_dis.setText('停止')

    # 修改像素比对阈值     
    def set_limit_value(self):
        self.limit_value = self.limit_value_set.value()
        self.limit_value_now.setText('{}'.format(self.limit_value))

    # 修改各通道波形检测的参数   
    def set_para_1(self):
        self.colorThresh[0] = self.ysyz_2.value()
        self.RGB[0] = self.ysxz_2.value()
        self.td1_yz.setText('{}'.format(self.colorThresh[0]))
        self.td1_ys.setText('{}'.format(self.RGB[0]))
        self.jcyz.setText('{}'.format(self.colorThresh[0]))
        self.jcys.setText('{}'.format(self.RGB[0]))

    def set_para_2(self):
        self.colorThresh[1] = self.ysyz_2.value()
        self.RGB[1] = self.ysxz_2.value()
        self.td2_yz.setText('{}'.format(self.colorThresh[1]))
        self.td2_ys.setText('{}'.format(self.RGB[1]))

    def set_para_3(self):
        self.colorThresh[2] = self.ysyz_2.value()
        self.RGB[2] = self.ysxz_2.value()
        self.td3_yz.setText('{}'.format(self.colorThresh[2]))
        self.td3_ys.setText('{}'.format(self.RGB[2]))

    def set_para_4(self):
        self.colorThresh[3] = self.ysyz_2.value()
        self.RGB[3] = self.ysxz_2.value()
        self.td4_yz.setText('{}'.format(self.colorThresh[3]))
        self.td4_ys.setText('{}'.format(self.RGB[3]))

    # 改变延时报警时间
    def alarm_time(self):
        self.th = int(self.alarm_time_sp.text())
        self.alarm_time_pv.setText('{} s'.format(self.th))
        
    # 结果显示开启与关闭
    def ocr_flag_xg(self):
        if not self.ocr_output_flag:
            self.ocr_output_flag = True
            self.ocr_dis.setText('显示关闭')
            self.wave_tab.raise_()
            self.txt_tab.raise_()
        else:
            self.ocr_output_flag = False
            #self.txt_tab.clear()
            self.ocr_dis.setText('显示开启')
            self.txt_tab.lower()
            self.wave_tab.lower()

    # 远程连接
    def remote_connect(self):
        # 创建socket
        self.tcp_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # IP 和端口
        server_ip = 'bemfa.com'
        server_port = 8344
        # 设置超时时间为5秒
        timeout = 0.5
        self.tcp_client_socket.settimeout(timeout)
        try:
            # 连接服务器
            self.tcp_client_socket.connect((server_ip, server_port))
            print("连接服务器成功！")
        except socket.error as e:
            print("Socket error:", e)

    # 更改服务器接受主题
    def control_recorder_(self):
        self.control_recorder = 0
    # 远程功能开启与关闭
    def remote_control(self):
        if self.remote.text() == '远程开启':
            self.remote_connect()
            self.remote_flag = True
            self.remote.setText('远程关闭')
        elif self.remote.text() == '远程关闭':
            self.tcp_client_socket.close()
            self.remote_flag = False
            self.control_change = False
            self.remote.setText('远程开启')

    # 添加阈值
    def Add_Th_list(self):  # 将置信值添加至list
        if len(th_max) >= self.thd_spinBox.value():
            th_max[self.thd_spinBox.value()-1] = self.th_max_spinBox.value()
            th_min[self.thd_spinBox.value()-1] = self.th_min_spinBox.value()
            th_name[self.thd_spinBox.value()-1] = self.th_name_spinBox.text() if self.th_name_spinBox.text() else '-'
            # print(self.thd_spinBox.value())
            # print(len(th_max))
            # print(th_max)
            # print(th_name)
            th_d[self.thd_spinBox.value()-1] = self.thd_spinBox.value()
        #print(self.th_name_spinBox.text())
        else:
            a = int(self.thd_spinBox.value())
            b = int(len(th_max))
            c = a-b-1
            for i in range(c):
                th_max.append('-')
                th_min.append('-')
                th_name.append('-')
                th_d.append('-')

            th_max.append(self.th_max_spinBox.value())
            th_min.append(self.th_min_spinBox.value())
            th_name.append(self.th_name_spinBox.text())
            th_d.append(self.thd_spinBox.value())

    # 数据保存函数
    def save_data(self,list_x,list_name, list_Max,list_r, list_Min, save_time, ala_s):   
        list_r = [re.sub("[^0-9,.,/]", " ", s) for s in list_r]
        title=["序号","名称","上限","实际值","下限"]

        if len(list_r) > len(list_Max):
            b = len(list_r) - len(list_Max) 
            for i in range(b):
                list_x.append("-")
                list_Max.append("-")
                list_Min.append("-")
                list_name.append("-")                
        elif len(list_r) < len(list_Max):
            a = len(list_Max) - len(list_r) 
            for i in range(a):
                list_r.append("-")

        save=[list_x,list_name,list_Max,list_r,list_Min]
        #save=np.array(save).reshape(5,len(list_x)) 
        save=list(map(list, zip(*save)))
        # print(save)
        time = QDateTime.currentDateTime()  # 获取当前时间
        time = time.toString("yyyy-MM-dd hh:mm:ss")  # 格式化一下时间
            
        save.insert(0,title)
        
        if ala_s:
            alarm = ["","","","","",time,"该组数据有异常！"]
            save.insert(0,alarm)
        else:
            time_lis=["","","","","",time]
            save.insert(0,time_lis)
        mid = pd.DataFrame(save)
        # print(self.data_path+'{0}.csv'.format(save_time))
        mid.to_csv(self.data_path+"\\"+'{0}.csv'.format(save_time), encoding="utf_8_sig", mode='a+', header=False, index=False)
    # 数据保存功能开启
    def save_open(self):
        if self.save_start.text() == '开始保存':
            self.saving = 1
            time = QDateTime.currentDateTime()  # 获取当前时间
            time = time.toString("yyyy-MM-dd hh时mm分ss秒")  # 格式化一下时间
            time=str(time)
            self.save_time = time
            self.save_start.setText('停止保存')
        elif self.save_start.text() == '停止保存':
            self.saving = 0
            # self.save_result = None
            self.save_start.setText('开始保存')

    # 开启报警
    def open_alarm(self):
                                                                 
        if self.alarm_stop:
            if self.color:   
                self.alarm_start.setText('关闭警报')
                self.alarm_stop = False
                global c
                c=0
            else:
                mesBox = QMessageBox()
                mesBox.setWindowTitle('提示')
                mesBox.setText('报警灯未连接！')
                mesBox.setIcon(QMessageBox.Information)
                mesBox.setStandardButtons(QMessageBox.Yes)
                mesBox.setStyleSheet("QPushButton:hover{background-color: rgb(255, 93, 52);}")
                mesBox.exec_()
            # if self.remote_flag:
            # sendTCP(self.tcp_client_socket, "alarm01","light_on")                                               
        else:
            self.color = send_light(command_close_all)
            if self.color:
                self.alarm_start.setText('打开警报')
                self.alarm_stop = True
            else:
                mesBox = QMessageBox()
                mesBox.setWindowTitle('提示')
                mesBox.setText('报警灯未连接！')
                mesBox.setIcon(QMessageBox.Information)
                mesBox.setStandardButtons(QMessageBox.Yes)
                mesBox.setStyleSheet("QPushButton:hover{background-color: rgb(255, 93, 52);}")
                mesBox.exec_()
            #更新远程线上状态
            # if self.remote_flag:
            # sendTCP(self.tcp_client_socket, "alarm01","light_off")   

    # 实时参数显示及历史报警显示
    def data_display(self,th_name,th_max,txt,th_min):
        self.data_dis.clearContents()
        txt = [re.sub("[^0-9,.,/]", "", s) for s in txt]

        items = [th_name,th_max,th_min]
        items = list(map(list, zip(*items)))
        self.data_dis.setRowCount(max(len(txt),len(th_max)))
        
        for i in range(len(items)):
            item = items[i]
            #row = self.data_dis.rowCount()
            #self.data_dis.insertRow(1)
            for j in range(len(item)):
                item = QTableWidgetItem(str(items[i][j]))
                item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

                self.data_dis.setItem(i,j,item)
        for i in range(len(txt)):
            item_txt = QTableWidgetItem(str(txt[i]))
            item_txt.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            if i in self.alarm_data and self.alarm_flag[i] == 0 :#报警记录显示判断
                self.alarm_display(th_name,th_max,txt,th_min,i)
            if i in self.alarm_data:
                item_txt.setBackground(QBrush(QColor(255,0,0)))  #设置背景色为红色
                self.alarm_flag[i] = 1
            else:
                self.alarm_flag[i] = 0
            self.data_dis.setItem(i,3,item_txt)


    # 远程app发送数据
    def send_data(self,th_d,th_Name,th_Max,th_txt,th_Min):
        th_txt = [re.sub("[^0-9,.,/]", "", s) for s in th_txt]
        if len(th_txt) > len(th_Max):
            b = len(th_txt) -len(th_Max) 
            for i in range(b):
                th_Max.append("-")
                th_Min.append("-")
                th_Name.append("-")
                th_d.append("-")

        elif len(th_txt) < len(th_Max):
            a = len(th_Max) - len(th_txt) 
            for i in range(a):
                th_txt.append("-")
        msg1 = "#"
        msg2 = "#"
        msg3 = "#"
        msg4 = "#"
        msg5 = "#"
        msg6 = '#'
        for i in range(len(th_Name)):
            msg1 = msg1+str(th_Name[i])+"#"
            msg2 = msg2+str(th_Max[i])+"#"
            msg3 = msg3+str(th_txt[i])+"#"
            msg4 = msg4+str(th_Min[i])+"#"

            if i in self.alarm_data:
                msg5 = msg5+"是"+"#"
            else:
                msg5 = msg5+"否"+"#"
            msg6 = "#"+str(len(th_Name))+"#"

        teamsendTCP(self.tcp_client_socket, "name","highdata","data01","lowdata","alarm","num","led","{}".format(msg1),"{}".format(msg2),"{}".format(msg3),"{}".format(msg4),"{}".format(msg5),"{}".format(msg6),"{}".format(self.color))
    # 远程控制app状态更新
    def send_control_data(self, msg, value):
        if value == 0:
            sendTCP(self.tcp_client_socket, "alarm01", "{}_off".format(msg))
        elif value == 1:
            sendTCP(self.tcp_client_socket, "alarm01", "{}_on".format(msg))
                    
    # 报警记录显示
    def alarm_display(self,th_name,th_max,txt,th_min,i):
        txt = [re.sub("[^0-9,.,/]", " ", s) for s in txt]
        self.alarm_list.insertRow(0)
        #获取当前时间
        time = QDateTime.currentDateTime()  # 获取当前时间
        now_time = time.toString("yyyy-MM-dd hh:mm:ss")  # 格式化一下时间
        item_txt = QTableWidgetItem(str(th_name[i]))
        item_txt.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.alarm_list.setItem(0,0,item_txt)

        item_txt = QTableWidgetItem(str(th_max[i]))
        item_txt.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.alarm_list.setItem(0,1,item_txt)

        item_txt = QTableWidgetItem(str(th_min[i]))
        item_txt.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.alarm_list.setItem(0,2,item_txt)

        item_txt = QTableWidgetItem(str(txt[i]))
        item_txt.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.alarm_list.setItem(0,3,item_txt)

        item_txt = QTableWidgetItem(str(now_time))
        item_txt.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.alarm_list.setItem(0,4,item_txt)

    # 波形的占空比检测
    def zkbjc(self, img, L, ori_p, tab, res_tab, th, rgb, flag): # 图像 线 原点 显示板
        x1,y1,x2,y2 = L[0],L[1],L[2],L[3]
        x0,y0 = ori_p[0], ori_p[1]
        zkb,img,err= wave_detect((x1-x0),(y1-y0),(x2-x0),(y2-y0),th,rgb,img)
        self.conf1.setText(zkb[:4])
        self.bx_err.setText(err)
        #图片显示
        self.bx_display.clear()
        s = self.bx_display.size()
        img = cv2.resize(img, (int(s.width()//4*4), int(s.height()//4*4)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)   
        tab.setPixmap(QPixmap.fromImage(img))
        tab.show()
        res_tab.setText(zkb)
        if not flag:
            self.bx_display.setPixmap(QPixmap.fromImage(img))
            self.bx_display.show()
        return(zkb)

    def path_change(self, save_p):
        if save_p == 'v':
            directory = QFileDialog.getExistingDirectory(None,"选取文件夹",self.video_path)  # 起始路径
            self.curr_video_dir.setText(directory)
            self.video_path = directory
        elif save_p == 'd':
            directory = QFileDialog.getExistingDirectory(None,"选取文件夹",self.data_path)  # 起始路径
            self.curr_data_dir.setText(directory)
            self.data_path = directory
        elif save_p == 'i':
            directory = QFileDialog.getExistingDirectory(None,"选取文件夹",self.pic_path)  # 起始路径
            self.curr_img_dir.setText(directory)
            self.pic_path = directory
    # 主循环
    def Display(self):
        j = 0 # infering counter
        n = 3 # alarm counter 
        c = 0 # buffer counter
        b = 0 # saving counter
        num_rec_box = 0 # num of rec_box
        save_counter = 0

        txt = []
        zhan = []

        while self.cap.isOpened():
            ok, frame = self.cap.read()
            #frame = undistort(frame)
            if not ok: # 关闭事件置为未触发，清空显示label
                self.stopEvent.clear()
                self.DisplayLabel.clear()
                self.DisplayLabel.setText("实时监视区")
                self.text_rec = 0
                self.initial_win()
                
                break
            # frame = undistort(frame)
            if j < 5:
                j = j+1
            else:
                j = 0
            s = self.DisplayLabel.size()
            frame = cv2.resize(frame, (int(s.width()//4*4), int(s.height()//4*4)))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.DisplayLabel.key == 3:
                self.select_box.setText('删除区域')
                self.select_area.setText('四点区域')
                if self.DisplayLabel.edit_line:
                    self.startLine.setText('删除画线')
                self.startButton2.setEnabled(False)
                self.DisplayLabel.key = 2

            if self.DisplayLabel.key == 4:
                self.select_box.setText('框选区域')
                self.select_area.setText('四点区域')
                self.startLine.setText('开始画线')
                if self.DisplayLabel.box_mode == 'wave':
                    self.startLine.setEnabled(True)
                self.startButton2.setEnabled(True)
                self.DisplayLabel.key = 2

            if self.DisplayLabel.key == 6:
                self.select_area.setText('删除区域')
                self.select_box.setText('框选区域')
                self.startButton2.setEnabled(False)
                self.DisplayLabel.key = 2

            if self.DisplayLabel.key == 9: # 区域建议
                self.set_button([0,0,1,1])
                self.proposal_box.setText('确认')
                self.proposal_point.setText('撤销')
                proposal_area = frame[self.DisplayLabel.prop_box[1]:self.DisplayLabel.prop_box[3],
                                      self.DisplayLabel.prop_box[0]:self.DisplayLabel.prop_box[2]]
                box_list0 = det_ov(proposal_area)
                for box in box_list0:
                    for p in box:
                        p += [self.DisplayLabel.prop_box[0], self.DisplayLabel.prop_box[1]] # 还原至初始坐标
                    box = (box.reshape(-1,8)).tolist()
                    self.prop_rec_box.append(box[0])
                self.DisplayLabel.key = 2

            if self.DisplayLabel.prop_box: # 建议确认
                dot_rect(frame, 
                        (self.DisplayLabel.prop_box[0], self.DisplayLabel.prop_box[1]), 
                        (self.DisplayLabel.prop_box[2], self.DisplayLabel.prop_box[3]), 
                        [255,0,0], 
                        thickness=2) # 需要改变线性
                for i, box in enumerate(self.prop_rec_box):
                    if len(box) == 4:
                        cv2.rectangle(frame, 
                                    (box[0], box[1]), 
                                    (box[2], box[3]), 
                                    [255,0,0], 
                                    thickness=2, 
                                    lineType=cv2.LINE_AA)
                    elif len(box) == 8:
                        pts = point_sort(box)
                        cv2.polylines(frame, 
                                    [pts], True, 
                                    [255,0,0], 
                                    thickness=2, 
                                    lineType=cv2.LINE_AA)

            if self.DisplayLabel.key == 11:
                box_list1 = det_ov(frame)
                for box in box_list1:
                    box = box.astype(np.int16)
                    box = (box.reshape(-1,8)).tolist()
                    if isPointInRect(self.DisplayLabel.test_point[0],
                                     self.DisplayLabel.test_point[1],
                                     box[0]):
                        self.DisplayLabel.rec_box.append(box[0])
                        break
                self.DisplayLabel.key = 4
                self.proposal_point.setText('点击建议')
                self.DisplayLabel.setCursor(Qt.ArrowCursor)

            if self.DisplayLabel.box_mode != 'text':
                self.select_area.setEnabled(0)

            if self.select_area.text()==('删除区域') or self.select_area.text()==('取消'):
                self.set_button([0,1,0,0])
                self.search_box.setEnabled(0)
                self.clean_box.setEnabled(0)

            if self.select_box.text()==('删除区域') or self.select_box.text()==('取消'):
                self.set_button([1,0,0,0])
                self.search_box.setEnabled(0)
                self.clean_box.setEnabled(0)

            if self.select_box.text()==('框选区域') and self.select_area.text()==('四点区域') and self.proposal_point.text()==('点击建议')and self.proposal_box.text()==('区域建议'):
                if self.DisplayLabel.box_mode == 'text':    
                    self.set_button([1,1,1,1])
                self.clean_box.setEnabled(1)
                if not self.DisplayLabel.rec_box:
                    self.search_box.setEnabled(1)

            if len(self.DisplayLabel.line) == len(self.DisplayLabel.wave_box) != 0:
                self.Boxing_start.setEnabled(1)
            else:
                self.Boxing_start.setEnabled(0)

            if self.flag_search and self.word_sear: # 自动搜索区域
                box_list= det_ov(frame)
                for box in box_list:
                    box = box.astype(np.int16)
                    box = (box.reshape(-1,8)).tolist()
                    self.DisplayLabel.rec_box.append(box[0])
                self.flag_search = False

            if not j:
                txt = []
                zhan = []
                self.err_rate = []
                # if self.text_rec and self.rec: # 识别模式
                if self.text_rec and self.rec0: # 识别模式
                    ocr = [] 
                    out = None
                    if not self.DisplayLabel.rec_box:
                        self.rec_stop()
                    else:
                        ocr = to_detbox(self.DisplayLabel.rec_box)
                        sc, out = rec_ov(frame, ocr)
                        # print('out',out)
                        if out:
                            for i in range(len(out)):
                                txt.append(out[i])
                                if (sc[i]<0.6) or (out[i]==''):
                                        txt[i] = '/'
                    # print('txt', txt)
                if self.wave_mode:
                    wave_img = []
                    ori_point = []
                    for i, box in enumerate(self.DisplayLabel.wave_box):
                        wave_img.append(frame[box[1]:box[3],
                                        box[0]:box[2]])
                        ori_point.append([box[0], box[1]])

                    for i, (L, w, p) in enumerate(zip(self.DisplayLabel.line, wave_img, ori_point)):
                        zhi = self.zkbjc(w, L, p, self.wave_disp_tab[i], self.wave_txt_tab[i], self.colorThresh[i], self.RGB[i], flag=i)
                        zhan.append(str(zhi))
                            
                if self.pic_compare and self.DisplayLabel.compare_box:#图像对比模式
 
                    if (self.compare_img is not None) and (self.target_img is not None):

                        alarm, signature, self.err_rate = compare(self, self.target_img, self.compare_img, self.limit_value)
                        self.err_rate_dis.setText('{}'.format(self.err_rate))
                        if self.err_rate > 1.0 and self.compare_save_flag :
                            if not save_counter:
                                signature0 = cv2.cvtColor(signature, cv2.COLOR_BGR2RGB)
                                datet = datetime.datetime.now().strftime('%Y-%m-%d %H_%M_%S') #添加时间戳
                                cv2.imwrite(self.pic_path +"\\" +'{}.jpg'.format(datet),signature0)#以当前时间为照片对象
                                save_counter += 1
                            elif save_counter < 20:
                                save_counter += 1
                            else:
                                save_counter = 0

                        s3 = self.real_dis.size()
                        if self.target_img is not None:
                            img3 = cv2.resize(signature, (int(s3.width()//4*4), int(s3.height()//4*4)))
                        else:
                            img3 = cv2.resize(self.compare_img, (int(s3.width()//4*4), int(s3.height()//4*4)))
                        # img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
                        img3 = QImage(img3.data, img3.shape[1], img3.shape[0], QImage.Format_RGB888)             
                        self.real_dis.setPixmap(QPixmap.fromImage(img3))
                        self.real_dis.show()

                data = txt + zhan + [str(self.err_rate)]
                C_Val, self.alarm_data = compare_list_elements(data, th_max, th_min)

                self.data_display(th_name,th_max,data,th_min) # 实时显示参数和报警显示

                if self.remote_flag:
                    self.send_data(th_d,th_name,th_max,data,th_min) # 远程app发送数据
                    # 远程APP状态更新
                    if self.control_change != self.remote_flag or \
                        self.control_recording_change != self.control_recording or\
                        self.control_saving_change != self.saving or\
                        self.alarm_stop_change != self.alarm_stop:
                        if self.saving == 1 and self.alarm_stop: 
                            self.send_control_data('alarm_off_save_on_and_recorder',self.control_recording)
                        elif self.saving == 0 and self.alarm_stop:
                            self.send_control_data('alarm_off_save_off_and_recorder', self.control_recording)
                        elif self.saving == 1 and not self.alarm_stop:
                            self.send_control_data('alarm_on_save_off_and_recorder', self.control_recording)
                        elif self.saving == 0 and not self.alarm_stop:
                            self.send_control_data('alarm_on_save_off_and_recorder', self.control_recording)
                        self.control_change = True
                        self.control_recording_change = self.control_recording
                        self.control_saving_change = self.saving
                        self.alarm_stop_change = self.alarm_stop
                        # self.control_count = 0

                    # 远程控制
                    control_alarm_flag, control_recording_flag, control_save_flag = 2, 2, 2
                    if self.control_recorder == 0:
                        control_alarm_flag, control_recording_flag, control_save_flag \
                            = receive_tcp(self.tcp_client_socket, 'alarm01')
                    
                    # 报警变量处理
                    if control_alarm_flag == 2:
                        pass
                    elif control_alarm_flag == 0:
                        if not self.alarm_stop :
                            self.open_alarm()                     
                        else:
                            pass
                    elif control_alarm_flag == 1:
                        if  self.alarm_stop and self.alarm_start.isEnabled():
                            self.open_alarm()
                        if  self.alarm_stop and not self.alarm_start.isEnabled():
                            sendTCP(self.tcp_client_socket, 'alarm01', 'alarm_off')  
                        else:
                            pass

                    # 录制变量处理
                    if control_recording_flag == 2:
                        pass
                    elif control_recording_flag == 0:
                        if self.control_recording == 1:
                            self.start_video()
                            self.control_recorder = 1
                        elif self.control_recording == 0:
                            pass
                    elif control_recording_flag == 1:
                        if self.control_recording == 0:
                            self.start_video()
                        elif self.control_recording == 1:
                            pass 
                    
                    # 保存变量处理
                    if control_save_flag == 2:
                        pass
                    elif control_save_flag == 0:
                        if self.saving == 1 and self.save_start.isEnabled():
                            self.save_open()
                        else:
                            pass

                    elif control_save_flag == 1:
                        if self.saving == 0 and self.save_start.isEnabled():
                            self.save_open()
                        elif self.saving == 0 and not self.save_start.isEnabled():
                            sendTCP(self.tcp_client_socket, 'alarm01', 'save_off')
                        else:
                            pass

                    if self.control_recorder == 1:
                        control_recorder_data = receive_tcp(self.tcp_client_socket, 'othercontrol')
                        if control_recorder_data == 1:
                            self.over_record.qout()
                            self.control_recorder = 0           

                if self.color == "close" and self.alarm_stop == 1:
                    self.color=send_light(command_green)
                
                if C_Val == 0:
                    n = n+1
                    if n == 4:
                        # self.v_lable.setText("异常")
                        # self.led_label.setStyleSheet("background-color: rgb(255, 0, 0);\n"
                        #                         "border-radius: 17px;\n"
                        #                         "border: 10px;")
                        # 报警后即时保存
                        if self.saving == 1:
                            save_time=self.save_time
                            self.alarm_save = True
                            self.save_data(th_d,th_name,th_max,data,th_min,save_time,self.alarm_save)
                        
                        if self.color == "close" and self.alarm_stop == 1:
                            self.color=send_light(command_green)

                        if self.alarm_stop == 0 and self.color == "green":
                            self.color=send_light(command_yellow)
                            c = 0
                            
                        if self.alarm_stop == 0 and self.color == "yellow":
                            if c < self.th:
                                c = c+1
                            else:
                                self.color = send_light(command_red)
                                c = 0
                        n = 0

                elif self.alarm_stop == 0 and self.color != "red":
                    n = 0
                    if c > 0:
                        c-=1
                    else:
                        # self.v_lable.setText("正常")
                        self.color=send_light(command_green)
                        # self.led_label.setStyleSheet("background-color: rgb(0, 255, 0);\n"
                        #                         "border-radius: 17px;\n"
                        #                         "border: 10px;")
                        self.alarm_save = False
                    
            if self.ocr_output_flag:
                self.ocr_output(txt)
                self.zhi_output(zhan, len(self.DisplayLabel.rec_box))

            frame = number_ocr(frame, self.DisplayLabel.edit_box, 1, self.DisplayLabel.edit_id)
            
            if self.DisplayLabel.rec_box: # 标注预测框为红色
                for i, box in enumerate(self.DisplayLabel.rec_box):
                    if len(box) == 4:
                        cv2.rectangle(frame, 
                                    (box[0], box[1]), 
                                    (box[2], box[3]), 
                                    [255,0,0], 
                                    thickness=2, 
                                    lineType=cv2.LINE_AA)
                    elif len(box) == 8:
                        pts = point_sort(box)
                        cv2.polylines(frame, 
                                    [pts], True, 
                                    [255,0,0], 
                                    thickness=2, 
                                    lineType=cv2.LINE_AA)

                frame = number_ocr(frame, self.DisplayLabel.rec_box, 0)
            else:
                self.startButton2.setEnabled(False)

            if self.DisplayLabel.wave_box: # 标注波形框为绿色
                for box in self.DisplayLabel.wave_box:
                    cv2.rectangle(frame, 
                                (box[0], box[1]), 
                                (box[2], box[3]), 
                                [0,255,0], 
                                thickness=2, 
                                lineType=cv2.LINE_AA)
                frame = number_ocr(frame, self.DisplayLabel.wave_box, 2, al_num=len(self.DisplayLabel.rec_box))
            else:
                self.Boxing_start.setEnabled(False)
                self.startLine.setEnabled(False)
                    
            if self.DisplayLabel.compare_box: # 标注比对框为棕色
                cv2.rectangle(frame, 
                            (self.DisplayLabel.compare_box[0], self.DisplayLabel.compare_box[1]), 
                            (self.DisplayLabel.compare_box[2], self.DisplayLabel.compare_box[3]), 
                            [0,0,0], 
                            thickness=2, 
                            lineType=cv2.LINE_AA)
                frame = number_ocr(frame, self.DisplayLabel.compare_box, 3, al_num=len(self.DisplayLabel.rec_box)+len(self.DisplayLabel.wave_box))
                self.compare_img =(frame[self.DisplayLabel.compare_box[1]:self.DisplayLabel.compare_box[3],
                                        self.DisplayLabel.compare_box[0]:self.DisplayLabel.compare_box[2]])
            if self.DisplayLabel.rec_area: # 绘制四边形连线
                if len(self.DisplayLabel.rec_area) >= 4:
                    for i in range(len(self.DisplayLabel.rec_area)//2-1):
                        cv2.line(frame,
                                (self.DisplayLabel.rec_area[2*i],self.DisplayLabel.rec_area[2*i+1]),
                                (self.DisplayLabel.rec_area[2*i+2],self.DisplayLabel.rec_area[2*i+3]),
                                [255,0,0], 
                                thickness=2, 
                                lineType=cv2.LINE_AA)
                        
            if self.DisplayLabel.line: # 波形检测占空比画线
                    for L in self.DisplayLabel.line:
                        cv2.line(frame,
                            (L[0],L[1]),
                            (L[2],L[3]),
                            [0,255,0], 
                            thickness=2, 
                            lineType=cv2.LINE_AA)
                            
            if self.DisplayLabel.edit_box: # 待删除框标蓝色
                if len(self.DisplayLabel.edit_box) == 4:
                    cv2.rectangle(frame, 
                                (self.DisplayLabel.edit_box[0], self.DisplayLabel.edit_box[1]), 
                                (self.DisplayLabel.edit_box[2], self.DisplayLabel.edit_box[3]), 
                                [0,0,255], 
                                thickness=2, 
                                lineType=cv2.LINE_AA)        
                elif len(self.DisplayLabel.edit_box) == 8:
                    pts_edit = point_sort(self.DisplayLabel.edit_box)
                    cv2.polylines(frame, 
                                [pts_edit], True, 
                                [0,0,255], 
                                thickness=2, 
                                lineType=cv2.LINE_AA)
                    
                if self.DisplayLabel.edit_line:
                    cv2.line(frame,
                                (self.DisplayLabel.edit_line[0],self.DisplayLabel.edit_line[1]),
                                (self.DisplayLabel.edit_line[2],self.DisplayLabel.edit_line[3]),
                                [0,0,255], 
                                thickness=2, 
                                lineType=cv2.LINE_AA)
                    
            #以下程序为记录保存程序
            if self.saving == 1 and self.alarm_save == False:
                if b<20:
                    b=b+1
                else:
                    # print('save!')
                    save_time=self.save_time
                    self.save_data(th_d,th_name,th_max,data,th_min,save_time,self.alarm_save)
                    b=0
            if self.recording == 1:
                if self.save_result:
                    datet = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') #添加时间戳
                    frame = Add_text(frame, "{}".format(datet), (1680, 25))
                    record = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # frame = Add_text(frame, 'Recording...', (20, 15))
                    record = cv2.resize(record, (1920,1080))
                    # record = cv2.resize(frame, (1280,960))

                    self.save_result.write(record)

            if self.recording == 2:
                self.save_result.release()
                self.recording = 0

            img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)             
            self.DisplayLabel.setPixmap(QPixmap.fromImage(img))
            self.DisplayLabel.show()
            
            # 判断关闭事件是否已触发
            if True == self.stopEvent.is_set(): # 关闭事件置为未触发，清空显示label
                self.stopEvent.clear()
                self.DisplayLabel.clear()
                self.DisplayLabel.setText("实时监视区")
                break
        self.cap.release()

    # 退出程序
    def quit_out(self):
        if self.recording == 1:
            self.recording = 0 # 结束视频录制
            self.over_record.show()

        time.sleep(0.5)
        self.shut_camera()
        self.win_ex.show()
        self.color = send_light(command_close_all) # 关闭报警灯
        QTimer.singleShot(1500, self.win_ex.close)
        time.sleep(1)
        self.close()

# 录制结束弹窗
class Dialog1(QDialog, Ui_Dialog1):
    def __init__(self):
        super(Dialog1, self).__init__()
        self.setupUi(self)
        self.setWindowIcon(QIcon('OnlineMonitoring/V4R.png'))
        self.setWindowTitle('录制结束')
        self.ok.clicked.connect(self.qout)
        self.check.clicked.connect(self.qout)

    def qout(self):
        self.close()

# 打开相机弹窗
class Dialog_op(QDialog, Ui_Dialog_op):
    def __init__(self):
        super(Dialog_op, self).__init__()
        self.setupUi(self)
        self.setWindowIcon(QIcon('OnlineMonitoring/V4R.png'))
        self.setWindowTitle('开启相机')

# 相机未连接
class Dialog_false(QDialog, Ui_Dialog_false):
    def __init__(self):
        super(Dialog_false, self).__init__()
        self.setupUi(self)
        self.setWindowIcon(QIcon('OnlineMonitoring/V4R.png'))
        self.setWindowTitle('开机失败')

# 关闭相机弹窗
class Dialog_sh(QDialog, Ui_Dialog_sh):
    def __init__(self):
        super(Dialog_sh, self).__init__()
        self.setupUi(self)
        self.setWindowIcon(QIcon('OnlineMonitoring/V4R.png'))
        self.setWindowTitle('关闭相机')

# 退出程序弹窗
class Dialog_ex(QDialog, Ui_Dialog_ex):
    def __init__(self):
        super(Dialog_ex, self).__init__()
        self.setupUi(self)
        self.setWindowIcon(QIcon('OnlineMonitoring/V4R.png'))
        self.setWindowTitle('退出')

def main():
    # authorize_serial_number = '1422723024529'
    # with open("/sys/firmware/devicetree/base/serial-number", "r") as f:
    #     serial_number = str(f.read().rstrip("\x00"))
    authorize_serial_number = '0000_0000_0000_0000_8CE3_8E03_0093_58E5.'
    for disk_drive in wmi.WMI().Win32_DiskDrive():
        serial_number = str(disk_drive.SerialNumber.rstrip("\x00"))
    if authorize_serial_number == serial_number:    
        app = QApplication([])
        stats = Stats()
        stats.show()
        stats.start_timer()
        app.exec_()
    else:
        print('应用程序未授权!')


if __name__ == '__main__':
    app = QApplication([])
    stats = Stats()

    #stats.showFullScreen()
    stats.show()
    stats.start_timer()   
    app.exec_()
