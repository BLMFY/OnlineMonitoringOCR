# -*- coding: utf-8
import socket
import time


def sendTCP(tcp_client_socket, topic,msg):

    substr1 = 'cmd=2&uid=3a7b47055d59428088e3dab077ba305a&topic={A}&msg={B}$\r\n'.format(A=topic,B=msg)#发布数据
    try:
        tcp_client_socket.send(substr1.encode("utf-8"))

    except:
        # print("远程发送失败,正在重新发送！")
        tcp_client_socket_connect('bemfa.com', 8344)
        tcp_client_socket.send(substr1.encode("utf-8"))

def teamsendTCP(tcp_client_socket,topic1,topic2,topic3,topic4,topic5,topic6,topic7,msg1,msg2,msg3,msg4,msg5,msg6,msg7):

    try:
        #发送订阅指令
        substr1 = 'cmd=2&uid=3a7b47055d59428088e3dab077ba305a&topic={A}/set&msg={B}$\r\n'.format(A=topic1,B=msg1)#发布数据
        substr2 = 'cmd=2&uid=3a7b47055d59428088e3dab077ba305a&topic={A}/set&msg={B}$\r\n'.format(A=topic2,B=msg2)#发布数据
        substr3 = 'cmd=2&uid=3a7b47055d59428088e3dab077ba305a&topic={A}/set&msg={B}$\r\n'.format(A=topic3,B=msg3)#发布数据
        substr4 = 'cmd=2&uid=3a7b47055d59428088e3dab077ba305a&topic={A}/set&msg={B}$\r\n'.format(A=topic4,B=msg4)#发布数据
        substr5 = 'cmd=2&uid=3a7b47055d59428088e3dab077ba305a&topic={A}/set&msg={B}$\r\n'.format(A=topic5,B=msg5)#发布数据
        substr6 = 'cmd=2&uid=3a7b47055d59428088e3dab077ba305a&topic={A}/set&msg={B}$\r\n'.format(A=topic6,B=msg6)#发布数据长度
        substr7 = 'cmd=2&uid=3a7b47055d59428088e3dab077ba305a&topic={A}/set&msg={B}$\r\n'.format(A=topic7,B=msg7)#发布现在的led颜色
        #comd = 'cmd=1&uid=5cd1c145abd84a3091b9c4151705a4ae&topic=alarm01\r\n'#发布订阅指令作为心跳
        
        
        tcp_client_socket.send(substr6.encode("utf-8"))#优先发送数据长度
        tcp_client_socket.send(substr1.encode("utf-8"))
        tcp_client_socket.send(substr2.encode("utf-8"))
        tcp_client_socket.send(substr3.encode("utf-8"))
        tcp_client_socket.send(substr4.encode("utf-8"))
        tcp_client_socket.send(substr5.encode("utf-8")) 
        
        tcp_client_socket.send(substr7.encode("utf-8"))

    except:
        tcp_client_socket_connect('bemfa.com', 8344)
        teamsendTCP(tcp_client_socket,topic1,topic2,topic3,topic4,topic5,topic6,topic7,msg1,msg2,msg3,msg4,msg5,msg6,msg7)
        # print("远程发送失败！")

def exchange(stop,rev):
    global recvData
    global rev_flag
    rev_flag = rev
    if "topic=alarm01" in recvData:
        if "msg=on" in recvData:
            alarm_stop = 0
            print("报警被远程开启！")
        elif 'msg=off' in recvData:
            alarm_stop = 1
            print("报警被远程关闭！")
    else:
        alarm_stop = stop
        print("报警状态保持不变，不被远程控制！")
    print(alarm_stop)
    return alarm_stop

# 远程连接
def tcp_client_socket_connect(ip, port):
    tcp_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # IP 和端口 
    server_ip = ip    #' bemfa.com'
    server_port = port  # 8344
    try:
        # 连接服务器
        tcp_client_socket.connect((server_ip, server_port))
        
    except:
        time.sleep(2)
        tcp_client_socket_connect(ip, port)
def receive_tcp(tcp_client_socket, topic):

    # 创建控制标志符号
    control_alarm_flag = 2
    control_recorder_flag = 2
    control_save_flag = 2
    control_recorder_ok = 2
    
    # 连接服务器
    if topic == 'othercontrol' or topic == 'alarm01':
        control_msg = 'cmd=3&uid=3a7b47055d59428088e3dab077ba305a&topic={}\r\n'.format(topic)

        try:
            tcp_client_socket.send(control_msg.encode("utf-8"))
        except:
            tcp_client_socket_connect('bemfa.com', 8344)
            tcp_client_socket.send(control_msg.encode("utf-8"))
    else:
        for i in topic:
            control_msg = 'cmd=3&uid=3a7b47055d59428088e3dab077ba305a&topic={}\r\n'.format(i)
            try:
                tcp_client_socket.send(control_msg.encode("utf-8"))
            except:
                tcp_client_socket_connect('bemfa.com', 8344)
                tcp_client_socket.send(control_msg.encode("utf-8"))

    control_data = tcp_client_socket.recv(1024)
    string_Data = control_data.decode('utf-8')
    list_string_Data = string_Data.split("\r\n")
    print("list:", list_string_Data)
    print("长度：", len(list_string_Data))

    # 录制弹窗控制 
    if topic == 'othercontrol':
        if string_Data.find('msg=ok') != -1 or string_Data.find('msg=check') != -1:
            control_recorder_ok = 1
        return control_recorder_ok
    else:
        # 报警控制
        if list_string_Data[1].find('msg=light_app_on') != -1:
            control_alarm_flag = 1
        elif list_string_Data[1].find('msg=light_app_off') != -1:
            control_alarm_flag = 0

        # 录制控制
        if list_string_Data[1].find('msg=recorder_app_on') != -1:
            control_recorder_flag = 1
        elif list_string_Data[1].find('msg=recorder_app_off') != -1:
            control_recorder_flag = 0
        # 保存控制
        if list_string_Data[1].find('msg=save_app_on') != -1:
            control_save_flag = 1
        elif list_string_Data[1].find('msg=save_app_off') != -1:
            control_save_flag = 0
        return control_alarm_flag, control_recorder_flag, control_save_flag

def Ping(tcp_client_socket):
    # 发送心跳
    try:
        keeplive = 'ping\r\n'
        tcp_client_socket.send(keeplive.encode("utf-8"))
    except:
        time.sleep(2)
        tcp_client_socket_connect('bemfa.com', 8344)
