'''
  Brought from 
  https://github.com/zju3dv/EasyMocap/blob/master/easymocap/mytools/utils.py
  https://github.com/zju3dv/EasyMocap/blob/master/easymocap/mytools/vis_base.py
'''
import os, sys
import time
import tabulate
import cv2
import numpy as np
import json
import importlib.util
import yaml
import copy
from ast import literal_eval
import open3d as o3d
from .create_meshes import create_mesh

def convert_verts_to_cam_space(vertices):
    # x - right, y - into, z - up
    vertices = vertices[:,[0,2,1]]
    vertices[:,2] *= -1
    vertices[:,1] -= 3
    return vertices

def get_uvs(uvmap_path):
    uv_map_vt_ft = np.load(uvmap_path, allow_pickle=True)
    vt, ft = uv_map_vt_ft['vt'], uv_map_vt_ft['ft']
    uvs = np.concatenate([vt[ft[:,ind]][:,None] for ind in range(3)],1).reshape(-1,2)
    uvs[:,1] = 1-uvs[:,1]
    return uvs

def load_sphere():
    cur_dir = os.path.dirname(__file__)
    faces = np.loadtxt(join(cur_dir, 'sphere_faces_20.txt'), dtype=np.int)
    vertices = np.loadtxt(join(cur_dir, 'sphere_vertices_20.txt'))
    return vertices, faces

def load_cylinder():
    cur_dir = os.path.dirname(__file__)
    faces = np.loadtxt(join(cur_dir, 'cylinder_faces_20.txt'), dtype=np.int)
    vertices = np.loadtxt(join(cur_dir, 'cylinder_vertices_20.txt'))
    return vertices, faces

def create_point_(points, r=0.01):
    """ create sphere
    Args:
        points (array): (N, 3)/(N, 4)
        r (float, optional): radius. Defaults to 0.01.
    """
    nPoints = points.shape[0]
    vert, face = load_sphere()
    nVerts = vert.shape[0]
    vert = vert[None, :, :].repeat(points.shape[0], 0)
    vert = vert + points[:, None, :]
    verts = np.vstack(vert)
    face = face[None, :, :].repeat(points.shape[0], 0)
    face = face + nVerts * np.arange(nPoints).reshape(nPoints, 1, 1)
    faces = np.vstack(face)
    return {'vertices': verts, 'faces': faces, 'name': 'points'}

def calRot(axis, direc):
    direc = direc/np.linalg.norm(direc)
    axis = axis/np.linalg.norm(axis)
    rotdir = np.cross(axis, direc)
    rotdir = rotdir/np.linalg.norm(rotdir)
    rotdir = rotdir * np.arccos(np.dot(direc, axis))
    rotmat, _ = cv2.Rodrigues(rotdir)
    return rotmat

def create_line_(start, end, r=0.01, col=None):
    length = np.linalg.norm(end[:3] - start[:3])
    vertices, faces = load_cylinder()
    vertices[:, :2] *= r
    vertices[:, 2] *= length/2    
    rotmat = calRot(np.array([0, 0, 1]), end - start)
    vertices = vertices @ rotmat.T + (start + end)/2
    ret = {'vertices': vertices, 'faces': faces, 'name': 'line'}
    if col is not None:
        ret['colors'] = col.reshape(-1, 3).repeat(vertices.shape[0], 0)
    return ret

def create_ground_(
    center=[0, 0, 0], xdir=[1, 0, 0], ydir=[0, 1, 0], # 位置
    step=1, xrange=10, yrange=10, # 尺寸
    white=[1., 1., 1.], black=[0.,0.,0.], # 颜色
    two_sides=True
    ):
    if isinstance(center, list):
        center = np.array(center)
        xdir = np.array(xdir)
        ydir = np.array(ydir)
    print('[Vis Info] {}, x: {}, y: {}'.format(center, xdir, ydir))
    xdir = xdir * step
    ydir = ydir * step
    vertls, trils, colls = [],[],[]
    cnt = 0
    min_x = -xrange if two_sides else 0
    min_y = -yrange if two_sides else 0
    for i in range(min_x, xrange):
        for j in range(min_y, yrange):
            point0 = center + i*xdir + j*ydir
            point1 = center + (i+1)*xdir + j*ydir
            point2 = center + (i+1)*xdir + (j+1)*ydir
            point3 = center + (i)*xdir + (j+1)*ydir
            if (i%2==0 and j%2==0) or (i%2==1 and j%2==1):
                col = white
            else:
                col = black
            vert = np.stack([point0, point1, point2, point3])
            col = np.stack([col for _ in range(vert.shape[0])])
            tri = np.array([[2, 3, 0], [0, 1, 2]]) + vert.shape[0] * cnt
            cnt += 1
            vertls.append(vert)
            trils.append(tri)
            colls.append(col)
    vertls = np.vstack(vertls)
    trils = np.vstack(trils)
    colls = np.vstack(colls)
    return {'vertices': vertls, 'faces': trils, 'colors': colls, 'name': 'ground'}


def get_rotation_from_two_directions(direc0, direc1):
    direc0 = direc0/np.linalg.norm(direc0)
    direc1 = direc1/np.linalg.norm(direc1)
    rotdir = np.cross(direc0, direc1)
    if np.linalg.norm(rotdir) < 1e-2:
        return np.eye(3)
    rotdir = rotdir/np.linalg.norm(rotdir)
    rotdir = rotdir * np.arccos(np.dot(direc0, direc1))
    rotmat, _ = cv2.Rodrigues(rotdir)
    return rotmat

PLANE_VERTICES = np.array([
    [0., 0., 0.],
    [1., 0., 0.],
    [0., 0., 1.],
    [1., 0., 1.],
    [0., 1., 0.],
    [1., 1., 0.],
    [0., 1., 1.],
    [1., 1., 1.]])
PLANE_FACES = np.array([
    [4, 7, 5],
    [4, 6, 7],
    [0, 2, 4],
    [2, 6, 4],
    [0, 1, 2],
    [1, 3, 2],
    [1, 5, 7],
    [1, 7, 3],
    [2, 3, 7],
    [2, 7, 6],
    [0, 4, 1],
    [1, 4, 5]], dtype=np.int32)

def create_plane(normal, center, dx=1, dy=1, dz=0.005, color=[0.8, 0.8, 0.8]):
    vertices = PLANE_VERTICES.copy()
    vertices[:, 0] = vertices[:, 0]*dx - dx/2
    vertices[:, 1] = vertices[:, 1]*dy - dy/2
    vertices[:, 2] = vertices[:, 2]*dz - dz/2
    # 根据normal计算旋转
    rotmat = get_rotation_from_two_directions(
        np.array([0, 0, 1]), np.array(normal))
    vertices = vertices @ rotmat.T
    vertices += np.array(center).reshape(-1, 3)
    return {'vertices': vertices, 'faces': PLANE_FACES.copy(), 'name': 'plane'}

def create_point(**kwargs):
    return create_mesh(**create_point_(**kwargs))

def create_line(**kwargs):
    return create_mesh(**create_line_(**kwargs))

def create_ground(**kwargs):
    ground = create_ground_(**kwargs)
    return create_mesh(**ground)

def create_coord(camera = [0,0,0], radius=1, scale=1):
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=radius, origin=camera)
    camera_frame.scale(scale)
    return camera_frame

def create_bbox(min_bound=(-3., -3., 0), max_bound=(3., 3., 2), flip=False):
    if flip:
        min_bound_ = min_bound.copy()
        max_bound_ = max_bound.copy()
        min_bound = [min_bound_[0], -max_bound_[1], -max_bound_[2]]
        max_bound = [max_bound_[0], -min_bound_[1], -min_bound_[2]]
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    return bbox


_YAML_EXTS = {"", ".yaml", ".yml"}
_PY_EXTS = {".py"}
def load_cfg(cfg_file_obj_or_str):
    """Load a cfg. Supports loading from:
        - A file object backed by a YAML file
        - A file object backed by a Python source file that exports an attribute
          "cfg" that is either a dict or a CN
        - A string that can be parsed as valid YAML
    """
    return _load_cfg_from_file(cfg_file_obj_or_str)

def _merge_a_into_b(a, b, root, key_list):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if '_no_merge_' in a.keys() and a['_no_merge_']:
        b.clear()
        a.pop('_no_merge_')

    for k, v_ in a.items():
        full_key = ".".join(key_list + [k])
        # a must specify keys that are in b
        if k not in b:
            if root.key_is_deprecated(full_key):
                continue
            elif root.key_is_renamed(full_key):
                root.raise_key_rename_error(full_key)
            else:
                v = copy.deepcopy(v_)
                v = _decode_cfg_value(v)
                b.update({k: v})
        else:
            v = copy.deepcopy(v_)
            v = _decode_cfg_value(v)
            v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, CN):
            try:
                _merge_a_into_b(v, b[k], root, key_list + [k])
            except BaseException:
                raise
        else:
            b[k] = v


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to CN objects
    if isinstance(v, dict):
        return CN(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, str):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    """Checks that `replacement`, which is intended to replace `original` is of
    the right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    original_type = type(original)
    replacement_type = type(replacement)

    # The types must match (with some exceptions)
    if replacement_type == original_type:
        return replacement

    # Cast replacement from from_type to to_type if the replacement and original
    # types match from_type and to_type
    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    # Conditionally casts
    # list <-> tuple
    casts = [(tuple, list), (list, tuple), (int, float), (float, int)]
    # For py2: allow converting from str (bytes) to a unicode string
    try:
        casts.append((str, unicode))  # noqa: F821
    except Exception:
        pass

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(
            original_type, replacement_type, original, replacement, full_key
        )
    )

def _load_cfg_from_file(file_obj):
    """Load a config from a YAML file or a Python source file."""
    _, file_extension = os.path.splitext(file_obj.name)
    if file_extension in _YAML_EXTS:
        return _load_cfg_from_yaml_str(file_obj.read())
    elif file_extension in _PY_EXTS:
        return _load_cfg_py_source(file_obj.name)
    else:
        raise Exception(
            "Attempt to load from an unsupported file type {}; "
            "only {} are supported".format(file_obj, _YAML_EXTS.union(_PY_EXTS))
        )


def _load_cfg_from_yaml_str(str_obj):
    """Load a config from a YAML string encoding."""
    cfg_as_dict = yaml.safe_load(str_obj)
    return CN(cfg_as_dict)


def _load_cfg_py_source(filename):
    """Load a config from a Python source file."""
    module = _load_module_from_file("yacs.config.override", filename)
    VALID_ATTR_TYPES = {dict, CN}
    if type(module.cfg) is dict:
        return CN(module.cfg)
    else:
        return module.cfg


# CfgNodes can only contain a limited set of valid types
_VALID_TYPES = {tuple, list, str, int, float, bool}
def _valid_type(value, allow_cfg_node=False):
    return (type(value) in _VALID_TYPES) or (allow_cfg_node and type(value) == CN)

class CN(dict):
    """
    CN represents an internal node in the configuration tree. It's a simple
    dict-like container that allows for attribute-based access to keys.
    """

    IMMUTABLE = "__immutable__"
    DEPRECATED_KEYS = "__deprecated_keys__"
    RENAMED_KEYS = "__renamed_keys__"

    def __init__(self, init_dict=None, key_list=None):
        # Recursively convert nested dictionaries in init_dict into CNs
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        for k, v in init_dict.items():
            if type(v) is dict:
                # Convert dict to CN
                init_dict[k] = CN(v, key_list=key_list + [k])
            else:
                # Check for valid leaf type or nested CN
                assert _valid_type(v, allow_cfg_node=True)
        super(CN, self).__init__(init_dict)
        # Manage if the CN is frozen or not
        self.__dict__[CN.IMMUTABLE] = False
        # Deprecated options
        # If an option is removed from the code and you don't want to break existing
        # yaml configs, you can add the full config key as a string to the set below.
        self.__dict__[CN.DEPRECATED_KEYS] = set()
        # Renamed options
        # If you rename a config option, record the mapping from the old name to the new
        # name in the dictionary below. Optionally, if the type also changed, you can
        # make the value a tuple that specifies first the renamed key and then
        # instructions for how to edit the config file.
        self.__dict__[CN.RENAMED_KEYS] = {
            # 'EXAMPLE.OLD.KEY': 'EXAMPLE.NEW.KEY',  # Dummy example to follow
            # 'EXAMPLE.OLD.KEY': (                   # A more complex example to follow
            #     'EXAMPLE.NEW.KEY',
            #     "Also convert to a tuple, e.g., 'foo' -> ('foo',) or "
            #     + "'foo:bar' -> ('foo', 'bar')"
            # ),
        }

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if self.is_frozen():
            raise AttributeError(
                "Attempted to set {} to {}, but CN is immutable".format(
                    name, value
                )
            )

        assert name not in self.__dict__
        assert _valid_type(value, allow_cfg_node=True)

        self[name] = value

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in self.items():
            seperator = "\n" if isinstance(v, CN) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 4)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super(CN, self).__repr__())

    def dump(self):
        """Dump to a string."""
        self_as_dict = _to_dict(self)
        return yaml.safe_dump(self_as_dict)

    def merge_from_file(self, cfg_filename):
        """Load a yaml config file and merge it this CN."""
        with open(cfg_filename, "r") as f:
            cfg = load_cfg(f)
        if 'parent' in cfg.keys():
            if cfg.parent != 'none':
                print('[Config] merge from parent file: {}'.format(cfg.parent))
                self.merge_from_file(cfg.parent)
        self.merge_from_other_cfg(cfg)

    def merge_from_other_cfg(self, cfg_other):
        """Merge `cfg_other` into this CN."""
        _merge_a_into_b(cfg_other, self, self, [])

    def merge_from_list(self, cfg_list):
        """Merge config (keys, values) in a list (e.g., from command line) into
        this CN. For example, `cfg_list = ['FOO.BAR', 0.5]`.
        """
        root = self
        for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
            if root.key_is_deprecated(full_key):
                continue
            if root.key_is_renamed(full_key):
                root.raise_key_rename_error(full_key)
            key_list = full_key.split(".")
            d = self
            for subkey in key_list[:-1]:
                d = d[subkey]
            subkey = key_list[-1]
            value = _decode_cfg_value(v)
            value = _check_and_coerce_cfg_value_type(value, d[subkey], subkey, full_key)
            d[subkey] = value

    def freeze(self):
        """Make this CN and all of its children immutable."""
        self._immutable(True)

    def defrost(self):
        """Make this CN and all of its children mutable."""
        self._immutable(False)

    def is_frozen(self):
        """Return mutability."""
        return self.__dict__[CN.IMMUTABLE]

    def _immutable(self, is_immutable):
        """Set immutability to is_immutable and recursively apply the setting
        to all nested CNs.
        """
        self.__dict__[CN.IMMUTABLE] = is_immutable
        # Recursively set immutable state
        for v in self.__dict__.values():
            if isinstance(v, CN):
                v._immutable(is_immutable)
        for v in self.values():
            if isinstance(v, CN):
                v._immutable(is_immutable)

    def clone(self):
        """Recursively copy this CN."""
        return copy.deepcopy(self)

    def register_deprecated_key(self, key):
        """Register key (e.g. `FOO.BAR`) a deprecated option. When merging deprecated
        keys a warning is generated and the key is ignored.
        """
        self.__dict__[CN.DEPRECATED_KEYS].add(key)

    def register_renamed_key(self, old_name, new_name, message=None):
        """Register a key as having been renamed from `old_name` to `new_name`.
        When merging a renamed key, an exception is thrown alerting to user to
        the fact that the key has been renamed.
        """
        value = new_name
        if message:
            value = (new_name, message)
        self.__dict__[CN.RENAMED_KEYS][old_name] = value

    def key_is_deprecated(self, full_key):
        """Test if a key is deprecated."""
        if full_key in self.__dict__[CN.DEPRECATED_KEYS]:
            print("Deprecated config key (ignoring): {}".format(full_key))
            return True
        return False

    def key_is_renamed(self, full_key):
        """Test if a key is renamed."""
        return full_key in self.__dict__[CN.RENAMED_KEYS]

    def raise_key_rename_error(self, full_key):
        new_key = self.__dict__[CN.RENAMED_KEYS][full_key]
        if isinstance(new_key, tuple):
            msg = " Note: " + new_key[1]
            new_key = new_key[0]
        else:
            msg = ""
        raise KeyError(
            "Key {} was renamed to {}; please update your config.{}".format(
                full_key, new_key, msg
            )
        )

class BaseConfig:
    @classmethod
    def load_from_args(cls):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg', type=str, default='config/vis/base.yml')
        parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
        args = parser.parse_args()
        return cls.load(filename=args.cfg, opts=args.opts)

    @classmethod
    def load(cls, filename=None, opts=[]) -> CN:
        cfg = CN()
        cfg = cls.init(cfg)
        if filename is not None:
            cfg.merge_from_file(filename)
        if len(opts) > 0:
            cfg.merge_from_list(opts)
        cls.parse(cfg)
        cls.print(cfg)
        return cfg
    
    @staticmethod
    def init(cfg):
        return cfg
    
    @staticmethod
    def parse(cfg):
        pass

    @staticmethod
    def print(cfg):
        print('[Info] --------------')
        print('[Info] Configuration:')
        print('[Info] --------------')
        print(cfg)

class Config(BaseConfig):
    @staticmethod
    def init(cfg):
        # input and output
        cfg.host = 'auto'
        cfg.port = 9999
        cfg.width = 1920
        cfg.height = 1080
        
        cfg.max_human = 5
        cfg.track = True
        cfg.block = True # block visualization or not, True for visualize each frame, False in realtime applications
        cfg.rotate = False
        cfg.debug = False
        cfg.write = False
        cfg.out = '/'
        # scene:
        cfg.scene_module = "easymocap.visualize.o3dwrapper"
        cfg.scene = CN()
        cfg.extra = CN()
        cfg.range = CN()
        cfg.new_frames = 0

        # skel
        cfg.skel = CN()
        cfg.skel.joint_radius = 0.02
        cfg.body_model_template = "none"
        # camera
        cfg.camera = CN()
        cfg.camera.phi = 0
        cfg.camera.theta = -90 + 60
        cfg.camera.cx = 0.
        cfg.camera.cy = 0.
        cfg.camera.cz = 6.
        cfg.camera.set_camera = False
        cfg.camera.camera_pose = []
        # range
        cfg.range = CN()
        cfg.range.minr = [-100, -100, -10]
        cfg.range.maxr = [ 100,  100,  10]
        cfg.range.rate_inlier = 0.8
        cfg.range.min_conf = 0.1
        return cfg
    
    @staticmethod
    def parse(cfg):
        if cfg.host == 'auto':
            cfg.host = socket.gethostname()
        if cfg.camera.set_camera:
            pass
        else:# use default camera
            # theta, phi = cfg.camera.theta, cfg.camera.phi
            theta, phi = np.deg2rad(cfg.camera.theta), np.deg2rad(cfg.camera.phi)
            cx, cy, cz = cfg.camera.cx, cfg.camera.cy, cfg.camera.cz
            st, ct = np.sin(theta), np.cos(theta)
            sp, cp = np.sin(phi), np.cos(phi)
            dist = 6
            camera_pose = np.array([
                    [cp, -st*sp, ct*sp, cx],
                    [sp, st*cp, -ct*cp, cy],
                    [0., ct, st, cz],
                    [0.0, 0.0, 0.0, 1.0]])
            cfg.camera.camera_pose = camera_pose.tolist()

class Timer:
    records = {}
    tmp = None

    @classmethod
    def tic(cls):
        cls.tmp = time.time()
    @classmethod
    def toc(cls):
        res = (time.time() - cls.tmp) * 1000
        cls.tmp = None
        return res
    
    @classmethod
    def report(cls):
        header = ['', 'Time(ms)']
        contents = []
        for key, val in cls.records.items():
            contents.append(['{:20s}'.format(key), '{:.2f}'.format(sum(val)/len(val))])
        print(tabulate.tabulate(contents, header, tablefmt='fancy_grid'))
    
    def __init__(self, name, silent=False):
        self.name = name
        self.silent = silent
        if name not in Timer.records.keys():
            Timer.records[name] = []
    
    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        end = time.time()
        Timer.records[self.name].append((end-self.start)*1000)
        if not self.silent:
            t = (end - self.start)*1000
            if t > 1000:
                print('-> [{:20s}]: {:5.1f}s'.format(self.name, t/1000))
            elif t > 1e3*60*60:
                print('-> [{:20s}]: {:5.1f}min'.format(self.name, t/1e3/60))
            else:
                print('-> [{:20s}]: {:5.1f}ms'.format(self.name, (end-self.start)*1000))


class BaseCrit:
    def __init__(self, min_conf, min_joints=3) -> None:
        self.min_conf = min_conf
        self.min_joints = min_joints
        self.name = self.__class__.__name__

    def __call__(self, keypoints3d, **kwargs):
        # keypoints3d: (N, 4)
        conf = keypoints3d[..., -1]
        conf[conf<self.min_conf] = 0
        idx = keypoints3d[..., -1] > self.min_conf
        return len(idx) > self.min_joints

class CritRange(BaseCrit):
    def __init__(self, minr, maxr, rate_inlier, min_conf) -> None:
        super().__init__(min_conf)
        self.min = minr
        self.max = maxr
        self.rate = rate_inlier
    
    def __call__(self, keypoints3d, **kwargs):
        idx = keypoints3d[..., -1] > self.min_conf
        k3d = keypoints3d[idx, :3]
        crit = (k3d[:, 0] > self.min[0]) & (k3d[:, 0] < self.max[0]) &\
        (k3d[:, 1] > self.min[1]) & (k3d[:, 1] < self.max[1]) &\
        (k3d[:, 2] > self.min[2]) & (k3d[:, 2] < self.max[2])
        self.log = '{}: {}'.format(self.name, k3d)
        return crit.sum()/crit.shape[0] > self.rate

def generate_colorbar(N = 20, cmap = 'jet'):
    bar = ((np.arange(N)/(N-1))*255).astype(np.uint8).reshape(-1, 1)
    colorbar = cv2.applyColorMap(bar, cv2.COLORMAP_JET).squeeze()
    if False:
        colorbar = np.clip(colorbar + 64, 0, 255)
    import random
    random.seed(666)
    index = [i for i in range(N)]
    random.shuffle(index)
    rgb = colorbar[index, :]
    rgb = rgb.tolist()
    return rgb

colors_bar_rgb = generate_colorbar(cmap='hsv')

colors_table = {
    'b': [0.65098039, 0.74117647, 0.85882353],
    '_pink': [.9, .7, .7],
    '_mint': [ 166/255.,  229/255.,  204/255.],
    '_mint2': [ 202/255.,  229/255.,  223/255.],
    '_green': [ 153/255.,  216/255.,  201/255.],
    '_green2': [ 171/255.,  221/255.,  164/255.],
    'r': [ 251/255.,  128/255.,  114/255.],
    '_orange': [ 253/255.,  174/255.,  97/255.],
    'y': [ 250/255.,  230/255.,  154/255.],
    '_r':[255/255,0,0],
    'g':[0,255/255,0],
    '_b':[0,0,255/255],
    'k':[0,0,0],
    '_y':[255/255,255/255,0],
    'purple':[128/255,0,128/255],
    'smap_b':[51/255,153/255,255/255],
    'smap_r':[255/255,51/255,153/255],
    'smap_b':[51/255,255/255,153/255],
}

def get_rgb(index):
    if isinstance(index, int):
        if index == -1:
            return (255, 255, 255)
        if index < -1:
            return (0, 0, 0)
        col = colors_bar_rgb[index%len(colors_bar_rgb)]
    else:
        col = colors_table.get(index, (1, 0, 0))
        col = tuple([int(c*255) for c in col[::-1]])
    return col

def get_rgb_01(index):
    col = get_rgb(index)
    return [i*1./255 for i in col[:3]]