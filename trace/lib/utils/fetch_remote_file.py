import os
import paramiko

local_cache_folder = '/home/yusun/Downloads/server_cacher'
server_url = {18:'10.207.174.18'}
def fetch_remote_file(remote_path, server_id=18):
    transport = paramiko.Transport((server_url[server_id], 22))
    transport.connect(username='suny', password='199497-')
    sftp = paramiko.SFTPClient.from_transport(transport)
    local_save_path = os.path.join(local_cache_folder, os.path.basename(remote_path))
    sftp.get(remote_path,local_save_path)
    return local_save_path

class Remote_server_fetcher(object):
    def __init__(self, server_id=18) -> None:
        super().__init__()
        transport = paramiko.Transport((server_url[server_id], 22))
        transport.connect(username='suny', password='199497-')
        self.sftp = paramiko.SFTPClient.from_transport(transport)
        self.local_cache_folder = '/home/yusun/Downloads/server_cacher'
    
    def fetch(self, remote_path):
        remote_path = '/home/sunyu15/datasets/3DPW/imageFiles/courtyard_arguing_00/image_00000.jpg'
        local_save_path = os.path.join(self.local_cache_folder, os.path.basename(remote_path))
        self.sftp.get(remote_path,local_save_path)
        return local_save_path

if __name__ == '__main__':
    RF = Remote_server_fetcher()
    RF.fetch('1')