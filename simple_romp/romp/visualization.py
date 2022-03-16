import cv2
import numpy as np
from sim3drender import RenderPipeline

"""
Brought from Tian Jin's https://pypi.org/project/realrender/
"""

def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order="C")
    return arr

"""
predefined config to render human mesh
"""

_cfg = {
    "intensity_ambient": .9,
    "intensity_directional": 0.6,
    "color_directional": (1, 1, 1),
    "intensity_specular": 0.2,
    "specular_exp": 5,
    "light_pos": (0, 0, 5),
    "view_pos": (0, 0, 5),
}
_render_app = RenderPipeline(**_cfg)

def render_human_mesh(
    img,
    ver_lst,
    tri,
    alpha=0.8,
    color=np.array([[0.65, 0.74, 0.86]]), 
):
    results = img.copy()
    for ver_ in ver_lst:
        ver_[:,2] *= -1
        ver = _to_ctype(ver_)  # transpose
        results = _render_app.render(ver, tri, results, color)
    results = cv2.addWeighted(img, 1 - alpha, results, alpha, 0)
    return results