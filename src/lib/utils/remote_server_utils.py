import pickle
import time
from multiprocessing.connection import Listener
import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
import config
import constants
from config import args


class Server_port_receiver(object):
    def __init__(self):
        host = args().server_ip
        port = args().server_port
        self.server_sock = Listener((host, port))
        self.conn = self.server_sock.accept()
    def receive(self):
        print('Server Listening')
        while True:
            data_bytes = self.conn.recv()
            data = pickle.loads(data_bytes)
            print('Received:', type(data))
            return data

    def send(self, data):
        data_bytes = pickle.dumps(data)
        self.conn.send(data_bytes)

if __name__ == '__main__':
    server_host = 'xxx.xxx.xxx.xxx'  # host = 'localhost'
    server_port = 10086  # if [Address already in use], use another port

    run_server(server_host, server_port)  # first, run this function only in server
    # run_client(server_host, server_port)  # then, run this function only in client
    pass
