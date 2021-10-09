import sys, os
root_dir = os.path.join(os.path.dirname(__file__),'..')
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
import socket
import time
from visualization.socket_utils import BaseSocketClient

def send_rand(client):
    import numpy as np
    N_person = 4
    datas = []
    for i in range(N_person):
        transl = (np.random.rand(1, 3) - 0.5) * 3
        kpts = np.random.rand(25, 4)
        kpts[:, :3] += transl
        pose = np.random.rand(1,72)
        beta = np.random.rand(1,10)
        data = {
            'id': i,
            'keypoints3d': kpts,
            'poses': pose,
            'betas': beta
        }
        datas.append(data)
    for _ in range(1):
        for i in range(N_person):
            move = (np.random.rand(1, 3) - 0.5) * 0.1
            datas[i]['keypoints3d'][:, :3] += move
        client.send_smpl(datas)
        time.sleep(0.01)
    client.close()

def send_dir(client, path, step):
    from os.path import join
    from glob import glob
    from tqdm import tqdm
    from easymocap.mytools.reader import read_keypoints3d
    results = sorted(glob(join(path, '*.json')))
    for result in tqdm(results[::step]):
        if args.smpl:
            data = read_smpl(result)
            client.send_smpl(data)
        else:
            data = read_keypoints3d(result)
            client.send(data)
        time.sleep(0.005)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9999)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--smpl', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if args.host == 'auto':
        args.host = socket.gethostname()
    for i in range(1000):
        client = BaseSocketClient(args.host, args.port)
        send_rand(client)