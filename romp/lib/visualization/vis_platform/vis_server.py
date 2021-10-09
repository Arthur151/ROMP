import open3d as o3d
version = int(o3d.__version__.split('.')[1])
if version==9:
    print('using open3d 0.9.0, importing functions from vis_server_py36_o3d9.')
    from .vis_server_py36_o3d9 import *
    config_file = 'romp/lib/visualization/vis_cfgs/o3d_scene_py36_o3d9.yml'
elif version >=11:
    print('using open3d 0.13.0, importing functions from vis_server_o3d13.')
    from .vis_server_o3d13 import *
    config_file = 'romp/lib/visualization/vis_cfgs/o3d_scene_o3d13.yml'
else:
    print('Error: the open3d version may not be supported.')

if __name__ == '__main__':
    cfg = Config().load(config_file)
    server = VisOpen3DSocket(cfg.host, cfg.port, cfg)
    while True:
        server.update()