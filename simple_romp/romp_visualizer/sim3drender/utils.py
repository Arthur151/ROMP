import numpy as np

def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order="C")
    return arr

def norm_vertices(vertices):
    vertices -= vertices.min(0)[None, :]
    vertices /= vertices.max()
    vertices *= 2
    vertices -= vertices.max(0)[None, :] / 2
    return vertices

def convert_type(obj):
    if isinstance(obj, tuple) or isinstance(obj, list):
        return np.array(obj, dtype=np.float32)[None, :]
    return obj

_norm = lambda arr: arr / np.sqrt(np.sum(arr**2, axis=1))[:, None]