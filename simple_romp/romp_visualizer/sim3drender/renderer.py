# coding: utf-8

import numpy as np
import Sim3DR_Cython
from .utils import _to_ctype, norm_vertices, convert_type, _norm

"""
Modified from Tian Jin's https://pypi.org/project/realrender/
"""

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
        self.intensity_ambient = convert_type(kwargs.get("intensity_ambient", 0.9))
        self.intensity_directional = convert_type(kwargs.get("intensity_directional", 0.6))
        self.intensity_specular = convert_type(kwargs.get("intensity_specular", 0.2))
        self.specular_exp = kwargs.get("specular_exp", 5)
        # self.color_ambient = convert_type(kwargs.get('color_ambient', (1, 1, 1)))
        self.color_directional = convert_type(kwargs.get('color_directional', (1, 1, 1)))
        self.light_pos = convert_type(kwargs.get("light_pos", (0, 0, 5)))
        self.view_pos = convert_type(kwargs.get("view_pos", (0, 0, 5)))

    def update_light_pos(self, light_pos):
        self.light_pos = convert_type(light_pos)

    def render(self, vertices, triangles, bg, color=np.array([[0.66, 0.5, 1.0]]), texture=None):
        normal = get_normal(vertices, triangles)

        # 2. lighting
        light = np.zeros_like(vertices, dtype=np.float32)
        # ambient component
        if self.intensity_ambient > 0:
            light += self.intensity_ambient * color

        vertices_n = norm_vertices(vertices.copy())
        if self.intensity_directional > 0:
            # diffuse component
            direction = _norm(self.light_pos - vertices_n)
            cos = np.sum(normal * direction, axis=1)[:, None]
            light += self.intensity_directional * (self.color_directional * np.clip(cos, 0, 1))

            # specular component
            if self.intensity_specular > 0:
                v2v = _norm(self.view_pos - vertices_n)
                reflection = 2 * cos * normal - direction
                spe = np.sum((v2v * reflection) ** self.specular_exp, axis=1)[:, None]
                spe = np.where(cos != 0, np.clip(spe, 0, 1), np.zeros_like(spe))
                light += self.intensity_specular * self.color_directional * np.clip(spe, 0, 1)
        
        light = np.clip(light, 0, 1)
        # light = np.ones_like(vertices, dtype=np.float32) * 0.5
        # 2. rasterization, [0, 1]
        if texture is None:
            render_img = rasterize(vertices, triangles, light, bg=bg)
            return render_img
        else:
            texture *= light
            render_img = rasterize(vertices, triangles, texture, bg=bg)
            return render_img

    def render_verts_list(self, verts_list, triangles, bg, color=np.array([[0.65, 0.74, 0.86]])):
        rendered_results = bg.copy()
        for verts in verts_list:
            verts[:,2] *= -1
            verts = _to_ctype(verts)
            rendered_results = self.render(verts, triangles, rendered_results, color)
        return rendered_results


def main():
    pass

if __name__ == "__main__":
    main()
