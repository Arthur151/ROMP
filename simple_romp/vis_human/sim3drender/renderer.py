# coding: utf-8

import numpy as np
import Sim3DR_Cython

"""
Brought from Tian Jin's https://pypi.org/project/realrender/
"""

def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order="C")
    return arr

_norm = lambda arr: arr / np.sqrt(np.sum(arr**2, axis=1))[:, None]


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

def get_normal(vertices, triangles):
    normal = np.zeros_like(vertices, dtype=np.float32)
    Sim3DR_Cython.get_normal(
        normal, vertices, triangles, vertices.shape[0], triangles.shape[0]
    )
    return normal

def rasterize(
    vertices,
    triangles,
    colors,
    bg=None,
    height=None,
    width=None,
    channel=None,
    reverse=False):
    if bg is not None:
        height, width, channel = bg.shape
    else:
        assert height is not None and width is not None and channel is not None
        bg = np.zeros((height, width, channel), dtype=np.uint8)

    buffer = np.zeros((height, width), dtype=np.float32) - 1e8

    if colors.dtype != np.float32:
        colors = colors.astype(np.float32)
    Sim3DR_Cython.rasterize(
        bg,
        vertices,
        triangles,
        colors,
        buffer,
        triangles.shape[0],
        height,
        width,
        channel,
        reverse=reverse,
    )
    return bg

class Sim3DR(object):
    def __init__(self, **kwargs):
        self.intensity_ambient = convert_type(kwargs.get("intensity_ambient", 0.66))
        self.intensity_directional = convert_type(kwargs.get("intensity_directional", 0.36))
        self.intensity_specular = convert_type(kwargs.get("intensity_specular", 0.1))
        self.specular_exp = kwargs.get("specular_exp", 1)
        self.color_directional = convert_type(kwargs.get("color_directional", (1, 1, 1)))
        self.light_pos = convert_type(kwargs.get("light_pos", (0, 0, -5)))
        self.view_pos = convert_type(kwargs.get("view_pos", (0, 0, 5)))

    def update_light_pos(self, light_pos):
        self.light_pos = convert_type(light_pos)

    def render(
        self, vertices, triangles, bg, color=np.array([[1, 0.6, 0.4]]), texture=None
    ):
        normal = get_normal(vertices, triangles)

        # 2. lighting
        light = np.zeros_like(vertices, dtype=np.float32)
        # ambient component
        if self.intensity_ambient > 0:
            light += self.intensity_ambient * np.array(color)

        vertices_n = norm_vertices(vertices.copy())
        if self.intensity_directional > 0:
            # diffuse component
            direction = _norm(self.light_pos - vertices_n)
            cos = np.sum(normal * direction, axis=1)[:, None]
            # cos = np.clip(cos, 0, 1)
            #  todo: check below
            light += self.intensity_directional * (
                self.color_directional * np.clip(cos, 0, 1)
            )

            # specular component
            if self.intensity_specular > 0:
                v2v = _norm(self.view_pos - vertices_n)
                reflection = 2 * cos * normal - direction
                spe = np.sum((v2v * reflection) ** self.specular_exp, axis=1)[:, None]
                spe = np.where(cos != 0, np.clip(spe, 0, 1), np.zeros_like(spe))
                light += (
                    self.intensity_specular
                    * self.color_directional
                    * np.clip(spe, 0, 1)
                )
        light = np.clip(light, 0, 1)

        # 2. rasterization, [0, 1]
        if texture is None:
            render_img = rasterize(vertices, triangles, light, bg=bg)
            return render_img
        else:
            texture *= light
            render_img = rasterize(vertices, triangles, texture, bg=bg)
            return render_img

    def __call__(self, verts_list, triangles, bg, mesh_colors=np.array([[1, 0.6, 0.4]])):
        rendered_results = bg.copy()
        if len(triangles.shape) == 2:
            triangles = [triangles for _ in range(len(verts_list))]
        for ind, verts in enumerate(verts_list):
            verts = _to_ctype(verts)
            rendered_results = self.render(verts, triangles[ind], rendered_results, mesh_colors[[ind%len(mesh_colors)]])
        return rendered_results


def main():
    pass

if __name__ == "__main__":
    main()
