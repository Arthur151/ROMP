import numpy as np
import pickle, random
import open3d as o3d
import os,sys

import config
import constants
from config import args
o3d_version = int(o3d.__version__.split('.')[1])

def get_uvs(uvmap_path):
    uv_map_vt_ft = np.load(uvmap_path, allow_pickle=True)
    vt, ft = uv_map_vt_ft['vt'], uv_map_vt_ft['ft']
    uvs = np.concatenate([vt[ft[:,ind]][:,None] for ind in range(3)],1).reshape(-1,2)
    uvs[:,1] = 1-uvs[:,1]
    return uvs

def parse_nvxia_uvmap(uvmap):
    uvs = uvmap[:,:,:2].reshape(-1,2)
    uvs[:,1] = 1-uvs[:,1]
    return uvs


def create_mesh_with_uvmap(vertices, faces, texture_path=None, uvs=None, **kwargs):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    if texture_path is not None and uvs is not None:
        if o3d_version==9:
            mesh.texture = o3d.io.read_image(texture_path)
            mesh.triangle_uvs = uvs
        elif o3d_version>=11:
            mesh.textures = [o3d.io.read_image(texture_path)]
            mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs)
            mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros((len(faces)), dtype=np.int32))
    mesh.compute_vertex_normals()
    return mesh


def create_mesh(vertices, faces, colors=None, **kwargs):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    if colors is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    else:
        mesh.paint_uniform_color([1., 0.8, 0.8])
    mesh.compute_vertex_normals()
    return mesh

def create_smpl_mesh():
    # if multi_mode:
    #     self.current_mesh_num = 10
    #     self.zero_vertices = o3d.utility.Vector3dVector(np.zeros((6890,3)))
    #     self.meshes = []
    #     for _ in range(self.current_mesh_num):
    #         new_mesh = self.create_smpl_mesh(self.verts_mean)
    #         self.meshes.append(new_mesh)
    #     self.set_meshes_zero(list(range(self.current_mesh_num)))
    
    smpl_param_dict = pickle.load(open(os.path.join(args().smpl_model_path,'SMPL_NEUTRAL.pkl'),'rb'), encoding='latin1')
    faces = smpl_param_dict['f']
    vertices = smpl_param_dict['v_template']
    if args().mesh_cloth in constants.wardrobe or args().mesh_cloth=='random':
        uvs = get_uvs(args().smpl_uvmap)
        if args().mesh_cloth=='random':
            mesh_cloth_id = random.sample(list(constants.wardrobe.keys()), 1)[0]
            print('choose mesh_cloth: ', mesh_cloth_id, constants.wardrobe[mesh_cloth_id])
            texture_file = os.path.join(args().wardrobe, constants.wardrobe[mesh_cloth_id])
        else:
            texture_file = os.path.join(args().wardrobe, constants.wardrobe[args().mesh_cloth])
        mesh = create_mesh_with_uvmap(vertices, faces, texture_path=texture_file, uvs=uvs)
    elif args().mesh_cloth in constants.mesh_color_dict:
        mesh_color = np.array(constants.mesh_color_dict[args().mesh_cloth])/255.
        mesh = create_mesh(vertices=vertices, faces=faces, colors=mesh_color)
    else:
        mesh = create_mesh(vertices=vertices, faces=faces)
    
    return mesh

def create_nvxia_mesh():
    params_path = os.path.join(args().nvxia_model_path, 'nvxia.npz')
    params_dict = np.load(params_path, allow_pickle=True)
    vertices = params_dict['coordinates']
    faces = np.array([np.array(face) for face in params_dict['polygons']])
    texture_file = os.path.join(args().nvxia_model_path, 'Kachujin_diffuse.png')
    uvs = parse_nvxia_uvmap(params_dict['uvmap'])
    mesh = create_mesh_with_uvmap(vertices, faces, texture_path=texture_file, uvs=uvs)
    return mesh

def create_body_mesh():
    if args().character == 'smpl':
        mesh = create_smpl_mesh()
        #self.view_mat = axangle2mat([1, 0, 0], np.pi) # align different coordinate systems
    elif args().character == 'nvxia':
        print('Loading NvXia model for visualization')
        mesh = create_nvxia_mesh()
    else:
        raise NotImplementedError
    return mesh

def create_body_model():
    if args().character == 'smpl':
        from models.smpl import SMPL
        model = SMPL(args().smpl_model_path)
    elif args().character == 'nvxia':
        from models.nvxia import create_nvxia_model
        model = create_nvxia_model(args().nvxia_model_path)
    else:
        raise NotImplementedError
    return model