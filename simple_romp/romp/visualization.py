import sys
import cv2
import numpy as np
import time

from sim3drender import RenderPipeline

def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order="C")
    return arr

"""
predefined config to render human mesh
"""

_cfg = {
    "intensity_ambient": .6,
    "intensity_directional": 0.6,
    "color_directional": (1, 1, 1),
    "intensity_specular": 0.1,
    "specular_exp": 5,
    "light_pos": (0, 0, 5),
    "view_pos": (0, 0, 5),
}
_render_app = RenderPipeline()#(**_cfg)

def render_single_image(
    img,
    ver_lst,
    tri,
    alpha=0.6,
    color=[0.56, 0.37, 0.96]):
    for ver_ in ver_lst:
        ver = _to_ctype(ver_)  # transpose
        overlap = _render_app.render(ver, tri, img.copy(), color)
        img = cv2.addWeighted(img, 1 - alpha, overlap, alpha, 0)

    return img

def render_human_mesh(
    img,
    ver_lst,
    tri,
    alpha=0.8,
    color=np.array([[0.56, 0.37, 0.96]]), #[0.98, 0.98, 0.84],#[0.65098039, 0.74117647, 0.85882353] ,#[0.56, 0.37, 0.96],
):
    results = img.copy()
    for ver_ in ver_lst:
        ver_[:,2] *= -1
        ver = _to_ctype(ver_)  # transpose
        results = _render_app.render(ver, tri, results, color)
    results = cv2.addWeighted(img, 1 - alpha, results, alpha, 0)
    return results