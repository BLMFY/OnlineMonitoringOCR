import serial
import binascii
import os


command_close_all=["01","05","00","00","00","00","CD","CA"]#全关
command_green = ["01","05","00","03","FF","00","7C","3A"]#绿灯亮
command_yellow=["01","05","00","1B","FF","00","FC","3D"]#黄灯亮+绿灯亮
command_red=["01","05","00","1A","FF","00","AD","FD"]#红灯亮

def get_light():
    lights = ['/dev/ttyCH341USB0', '/dev/ttyUSB0']
    for path in lights:
        if os.path.exists(path):
            return path
    # assert False,"can't find light"

def old_send_light(signal_light):
     print("报警开始")
     return ("green")
     
def send_light(signal_light):
    try:
        light_path = get_light()
        ser=serial.Serial(light_path, 9600,parity="E", stopbits=1, bytesize=8,timeout=0.5)#Linux系统使用ttyCH341USB0口连接串行口
        
        if signal_light == command_close_all:
            for i in command_close_all:
                byte_data = binascii.unhexlify(i)  # 转换为字节串
                
                ser.write(byte_data)  # 发送
                #ser.close()  # 关闭串口
            ser.close()
            return("close")
        
        if signal_light == command_red:
                for i in command_red:
                    byte_data = binascii.unhexlify(i)
                    ser.write(byte_data)
                    # print(byte_data)
                ser.close()
                return("red")
        elif signal_light == command_green:
                for i in command_green:
                    byte_data = binascii.unhexlify(i)
                    ser.write(byte_data)
                    # print(byte_data)
                ser.close()
                return("green")
        elif signal_light == command_yellow:
                for i in command_yellow:
                    byte_data = binascii.unhexlify(i)
                    #byte_data = command_yellow[i]
                    ser.write(byte_data)
                    # print(byte_data)
                ser.close()
                return("yellow")
    except:
         print("报警灯未连接")
         return False
#send_light(command_close_all)




