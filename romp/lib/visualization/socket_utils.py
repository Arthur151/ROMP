import socket
import time
from threading import Thread
from queue import Queue
import cv2
import numpy as np
import json

def myarray2string(array, separator=', ', fmt='%.3f', indent=8):
    assert len(array.shape) == 2, 'Only support MxN matrix, {}'.format(array.shape)
    blank = ' ' * indent
    res = ['[']
    for i in range(array.shape[0]):
        res.append(blank + '  ' + '[{}]'.format(separator.join([fmt%(d) for d in array[i]])))
        if i != array.shape[0] -1:
            res[-1] += ', '
    res.append(blank + ']')
    return '\r\n'.join(res)

def write_common_results(dumpname=None, results=[], keys=[], fmt='%2.3f'):
    format_out = {'float_kind':lambda x: fmt % x}
    out_text = []
    out_text.append('[\n')
    for idata, data in enumerate(results):
        out_text.append('    {\n')
        output = {}
        output['id'] = data['id']
        for key in keys:
            if key not in data.keys():continue
            # BUG: This function will failed if the rows of the data[key] is too large
            # output[key] = np.array2string(data[key], max_line_width=1000, separator=', ', formatter=format_out)
            output[key] = myarray2string(data[key], separator=', ', fmt=fmt)
        out_keys = list(output.keys())
        for key in out_keys:
            out_text.append('        \"{}\": {}'.format(key, output[key]))
            if key != out_keys[-1]:
                out_text.append(',\n')
            else:
                out_text.append('\n')
        out_text.append('    }')
        if idata != len(results) - 1:
            out_text.append(',\n')
        else:
            out_text.append('\n')
    out_text.append(']\n')
    if dumpname is not None:
        mkout(dumpname)
        with open(dumpname, 'w') as f:
            f.writelines(out_text)
    else:
        return ''.join(out_text)

def encode_detect(data):
    res = write_common_results(None, data, ['keypoints3d'])
    res = res.replace('\r', '').replace('\n', '').replace(' ', '')
    return res.encode('ascii')

def encode_smpl(data):
    res = write_common_results(None, data, ['poses', 'betas', 'vertices', 'transl'])
    res = res.replace('\r', '').replace('\n', '').replace(' ', '')
    return res.encode('ascii')

def encode_image(image):
    fourcc = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, img_encode = cv2.imencode('.jpg', image, fourcc)
    data = np.array(img_encode) # numpy array로 안바꿔주면 ERROR
    stringData = data.tostring()
    return stringData

def log(x):
    from datetime import datetime
    time_now = datetime.now().strftime("%m-%d-%H:%M:%S.%f ")
    print(time_now + x)

class BaseSocket:
    def __init__(self, host, port, debug=False) -> None:
        # 创建 socket 对象
        print('[Info] server start')
        serversocket = socket.socket(
                    socket.AF_INET, socket.SOCK_STREAM)
        serversocket.bind((host, port))
        serversocket.listen(1)
        self.serversocket = serversocket
        self.queue = Queue()
        self.t = Thread(target=self.run)
        self.t.start()
        self.debug = debug
        self.disconnect = False
    
    @staticmethod
    def recvLine(sock):
        flag = True
        result = b''
        while not result.endswith(b'\n'):
            res = sock.recv(1)
            if not res:
                flag = False
                break
            result += res
        return flag, result.strip().decode('ascii')

    @staticmethod
    def recvAll(sock, l):
        l = int(l)
        result = b''
        while (len(result) < l):
            t = sock.recv(l - len(result))
            result += t
        return result.decode('ascii')

    def run(self):
        while True:
            clientsocket, addr = self.serversocket.accept()
            print("[Info] Connect: %s" % str(addr))
            self.disconnect = False
            while True:
                flag, l = self.recvLine(clientsocket)
                if not flag:
                    print("[Info] Disonnect: %s" % str(addr))
                    self.disconnect = True
                    break
                data = self.recvAll(clientsocket, l)
                if self.debug:log('[Info] Recv data')
                self.queue.put(data)
            clientsocket.close()
    
    def update(self):
        time.sleep(1)
        while not self.queue.empty():
            log('update')
            data = self.queue.get()
            self.main(data)
    
    def main(self, datas):
        print(datas)

    def __del__(self):
        self.serversocket.close()
        self.t.join()


class BaseSocketClient:
    def __init__(self, host='127.0.0.1', port=9999) -> None:
        if host == 'auto':
            host = socket.gethostname()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        self.s = s
    
    def send(self, data):
        val = encode_detect(data)
        self.s.send(bytes('{}\n'.format(len(val)), 'ascii'))
        self.s.sendall(val)
    
    def send_smpl(self, data):
        val = encode_smpl(data)
        self.s.send(bytes("{}\n".format(len(val)), 'ascii'))
        self.s.sendall(val)
    
    def close(self):
        self.s.close()

class Results_sender():
    def __init__(self):
        self.client = BaseSocketClient()
        self.queue = Queue()
        self.t = Thread(target=self.run)
        self.t.start()

    def run(self):
        while True:
            time.sleep(1)
            while not self.queue.empty():
                log('update')
                data = self.queue.get()
                self.client.send_smpl(data)

    def send_results(self, poses=None, betas=None, verts=None, kp3ds=None, trans=None,ids=[]):
        
        results = []
        print('sending detected {} person results'.format(len(ids)))
        for ind, pid in enumerate(ids):
            result = {}
            result['id'] = pid
            # if kp3ds is not None:
            #     result['keypoints3d'] = kp3ds[[ind]]
            # if verts is not None:
            #     result['vertices'] = verts[[ind]]
            if trans is not None:
                result['transl'] = trans[[ind]]
            if poses is not None:
                result['poses'] = poses[[ind]]
            if betas is not None:
                result['betas'] = betas[[ind]]
            results.append(result)
        
        self.queue.put(results)

class SocketClient_blender:
    def __init__(self, host='127.0.0.1', port=9999) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((host, port))
        s.listen(1)
        print(f'bind on {port}')
        self.sock, addr = s.accept()
        self.s=s
    
    def send(self, data_list):
        d = self.sock.recv(1024)
        if not d:
            self.close()
        #data, addr = self.s.recvfrom(1024)
        data_send = json.dumps(data_list).encode('utf-8')
        self.sock.send(data_send)
    
    def close(self):
        self.s.close()

class SocketClient_blender_old:
    def __init__(self, host='127.0.0.1', port=9999) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind((host, port))
        print(f'bind on {port}')
        self.s = s
    
    def send(self, data_list):
        data, addr = self.s.recvfrom(1024)
        data_send = json.dumps(data_list)
        self.s.sendto(bytes(data_send.encode('utf-8')), addr)
    
    def close(self):
        self.s.close()
