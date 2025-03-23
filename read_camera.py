import cv2
import requests

i = 0
cap = cv2.VideoCapture('rtsp://admin:12345@192.168.1.68:554/ Stream/Live/101')  # 替换为视频路径或者设备索引（例如0表示摄像头）
zoom_level = 6000
# cap.set(cv2.CAP_PROP_ZOOM, zoom_level)
focus_level = 20000 #
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840) #2048
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160) #1536

def rtsp_login(ip_stream = 'http://192.168.1.68/SDK/UNIV_API'):
    payload = {"session": 0, "id":1, "call": {"service": "rpc", "method": "login"},\
                "params":{"userName":"admin", "password":"0b13ff306a543a373e9595543888d74c", "random": "310000",\
                        "ip":"192.168.1.1" , "port": 80, "encryptType":0}}
    response = requests.post(ip_stream, json=payload)
    session_number = (response.json())['params']['session']
    return session_number

def read_set(ip_stream = 'http://192.168.1.68/SDK/UNIV_API', f = 12306, z = 3242):
    session_number = rtsp_login(ip_stream)
    get_focus =  {"session": session_number, "id":2, "call": {"service": "ptz", "method": "getPTZPosParam"},\
            "params":{"channel":0,}}
    response = requests.post(' http://192.168.1.68/SDK/UNIV_API', json=get_focus)
    # print(response.json())
    if 'params' in response.json():
        f, z = (response.json())['params']['FocusPos'], (response.json())['params']['ZoomPos']
        print('f:', f, 'z:', z)
    return f, z

def zoom_set(mode, ip_stream = 'http://192.168.1.68/SDK/UNIV_API'):
    session_number = rtsp_login(ip_stream)
    get_focus =  {"session": session_number, "id":2, "call": {"service": "ptz", "method": "getPTZPosParam"},\
            "params":{"channel":0,}}
    fz = requests.post(' http://192.168.1.68/SDK/UNIV_API', json=get_focus)
    print(fz.json())
    if 'params' in fz.json():
        z = (fz.json())['params']['ZoomPos']
        if mode == '+':
            z += 500
            set_focus =  {"session": session_number, "id":2, "call": {"service": "ptz", "method": "setPTZPosParam"},\
            "params":{"channel": 0, "table":{"Action":6, "PanPos":1, "TiltPos":1, "ZoomPos":z, "FocusPos":1}}}
        elif mode == '-':
            z -= 500
            set_focus =  {"session": session_number, "id":2, "call": {"service": "ptz", "method": "setPTZPosParam"},\
            "params":{"channel": 0, "table":{"Action":6, "PanPos":1, "TiltPos":1, "ZoomPos":z, "FocusPos":1}}}
        response = requests.post(ip_stream, json=set_focus)
        print('z:', z)
    


def focus_set(mode, ip_stream = 'http://192.168.1.68/SDK/UNIV_API'):
    session_number = rtsp_login(ip_stream)
    get_focus =  {"session": session_number, "id":2, "call": {"service": "ptz", "method": "getPTZPosParam"},\
            "params":{"channel":0,}}
    fz = requests.post(' http://192.168.1.68/SDK/UNIV_API', json=get_focus)
    print(fz.json())
    if 'params' in fz.json():
        f = (fz.json())['params']['FocusPos']
        if mode == '+':
            f += 500
            set_focus =  {"session": session_number, "id":2, "call": {"service": "ptz", "method": "setPTZPosParam"},\
            "params":{"channel": 0, "table":{"Action":7, "PanPos":1, "TiltPos":1, "ZoomPos":1, "FocusPos":f}}}
        elif mode == '-':
            f -= 500
            set_focus =  {"session": session_number, "id":2, "call": {"service": "ptz", "method": "setPTZPosParam"},\
            "params":{"channel": 0, "table":{"Action":7, "PanPos":1, "TiltPos":1, "ZoomPos":1, "FocusPos":f}}}
        response = requests.post(ip_stream, json=set_focus)
        print('f:', f)

if __name__ == '__main__':
    # f,z = read_set()
    # zoom_set(zoom_level, ip_stream = 'http://192.168.1.68/SDK/UNIV_API')
    # focus_set(focus_level, ip_stream = 'http://192.168.1.68/SDK/UNIV_API')

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Frame', frame)

        if cv2.waitKey(3) & 0xFF == ord('a'):
            zoom_set('+', ip_stream = 'http://192.168.1.68/SDK/UNIV_API')

        if cv2.waitKey(3) & 0xFF == ord('d'):
            zoom_level -= 500
            zoom_set('-', ip_stream = 'http://192.168.1.68/SDK/UNIV_API')

        if cv2.waitKey(3) & 0xFF == ord('w'):
            focus_set('+', ip_stream = 'http://192.168.1.68/SDK/UNIV_API')

        if cv2.waitKey(3) & 0xFF == ord('s'):
            focus_set('-', ip_stream = 'http://192.168.1.68/SDK/UNIV_API')
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('quit')
            break
    
    cap.release()
    cv2.destroyAllWindows()
