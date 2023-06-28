import numpy as np
from PIL import Image
import plotly.graph_objects as go

import plotly
from plotly.graph_objs.scatter import Line
from plotly.subplots import make_subplots
import plotly.express as px
import ipywidgets
from ipywidgets.widgets import Layout, HBox, VBox
from ipywidgets.embed import embed_minimal_html
import pandas as pd
import sys, os, random
import constants
import config
from config import args  
import io, cv2
from PIL import Image

def plotly_fig2array(fig):
    #convert Plotly fig to  an array
    save_path = os.path.join('{}.png'.format(random.random()))
    fig.write_image(save_path,width=512,height=512)
    img = cv2.imread(save_path)
    os.remove(save_path)
    # fig_bytes = fig.to_image(format="png")
    # buf = io.BytesIO(fig_bytes)
    # img = Image.open(buf)
    return np.asarray(img)

def write_to_html(img_names, plot_dict, vis_cfg):
    containers = []
    raw_layout = Layout(overflow_x='scroll',border='2px solid black',width='1900px',height='',
                    flex_direction='row',display='flex')        
    for inds, img_name in enumerate(img_names):
        Hboxes = []
        for item in list(plot_dict.keys()):
            if inds >= len(plot_dict[item]['figs']):
                continue
            fig = plot_dict[item]['figs'][inds]
            fig['layout'] = {"title":{"text":img_name.replace(args().dataset_rootdir, '')}}
            Hboxes.append(go.FigureWidget(fig))
        containers.append(HBox(Hboxes,layout=raw_layout))
    all_figs = VBox(containers)
    save_name = os.path.join(vis_cfg['save_dir'],vis_cfg['save_name']+'.html')
    embed_minimal_html(save_name, views=[all_figs], title=vis_cfg['save_name'], drop_defaults=True)
    ipywidgets.Widget.close_all()
    del all_figs, containers, Hboxes

def prepare_coord_map(d, h, w):
    d_map = np.zeros((d,h,w))
    for ind in range(d):
        d_map[ind] = d-1-ind
    h_map = np.zeros((d,h,w))
    for ind in range(h):
        h_map[:,ind] = ind
    w_map = np.zeros((d,h,w))
    for ind in range(w):
        w_map[:,:,ind] = w-1-ind

    return [w_map, h_map, d_map]

whd_map = prepare_coord_map(64,128,128)

def add_image2fig(image, suf_w=128, suf_h=128):
    eight_bit_img = Image.fromarray(image[:,::-1]).convert('P', palette='WEB', dither=None)
    idx_to_color = np.array(eight_bit_img.getpalette()).reshape((-1, 3))
    colorscale=[[i/255.0, "rgb({}, {}, {})".format(*rgb)] for i, rgb in enumerate(idx_to_color)]

    h, w = image.shape[:2]
    x = np.linspace(1, suf_w, w)
    y = np.linspace(1, suf_h, h)
    z = np.zeros((h,w))

    image_fig = go.Surface(x=x, y=y, z=z,
        surfacecolor=eight_bit_img, 
        cmin=0, 
        cmax=255,
        colorscale=colorscale,
        showscale=False,
        lighting_diffuse=1,
        lighting_ambient=1,
        lighting_fresnel=1,
        lighting_roughness=1,
        lighting_specular=0.5,
        )
    return image_fig

def plot_3D_volume(volume):
    X, Y, Z = whd_map
    fig = go.Figure()
    
    volume_fig = go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        #isomin=-0.5,
        #isomax=1,
        # cmin=-0.1,
        # cmax=1.1,
        value=volume.flatten(),
        # degree of not transparent, the lower the value, the more transparent the volume would be.
        opacity=0.3, 
        # # the opacity values for different values, in [[value1, opacity], [value2, opacity], ...]
        #opacityscale=[[0, 0.1], [0.5, 0.3], [1, 0.5]], 
        opacityscale=[[-1, 0],[0, 0], [0.3, 1]],
        surface_count=10, # need to be large to show the target
        colorscale='RdBu'
        )
    return volume_fig

def update_centermap_layout(fig):
    fig.update_layout(
        title="CenterMap 3D",
        width=800,
        height=800,
        scene=dict( xaxis_visible=True,
                    yaxis_visible=True, 
                    zaxis_visible=True, 
                    xaxis_title="W",
                    yaxis_title="H",
                    zaxis_title="D"))
    return fig

def show_plotly_figure(volume=None, image=None):
    fig = go.Figure()
    if volume is not None:
        volume_fig = plot_3D_volume(volume)
        fig.add_trace(volume_fig)
    if image is not None:
        image_fig = add_image2fig(image)
        fig.add_trace(image_fig)
    fig = update_centermap_layout(fig)
    #fig.write_image("images/fig1.png")
    fig.show()
    #return plotly_fig2array(fig)

def plot_3dslice(volume, slice_thick=0.1):
    d, h, w = volume.shape

    fig = go.Figure(frames=[go.Frame(data=go.Surface(
        z=((d-1)*0.1 - k * slice_thick) * np.ones((h, w)),
        surfacecolor=np.flipud(volume[d-1 - k]),
        cmin=0, cmax=1
        ),
        name=str(k) # you need to name the frame for the animation to behave properly
        )
        for k in range(d)])

    # Add data to be displayed before animation starts
    fig.add_trace(go.Surface(
        z=(d-1)*0.1 * np.ones((h, w)),
        surfacecolor=np.flipud(volume[-1]),
        colorscale='Gray',
        cmin=0, cmax=1,
        colorbar=dict(thickness=20, ticklen=4)
        ))

    def frame_args(duration):
        return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }

    sliders = [{
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],
                            "label": str(k),
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],}]

    # Layout
    fig.update_layout(
            title='Slices in volumetric data',
            width=800,
            height=800,
            scene=dict(
                        zaxis=dict(range=[-slice_thick, d*0.1], autorange=False),
                        aspectratio=dict(x=1, y=1, z=1),
                        ),
            updatemenus = [
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(50)],
                            "label": "&#9654;", # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;", # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,}],
            sliders=sliders)

    fig.show()

def play_video_clips(frames): 
    fig = px.imshow(frames, animation_frame=0, binary_string=True)
    return fig
    fig.show()

def merge_figs2html(plotly_figs, save_path):
    for plotly_fig in plotly_figs:
        fig = px.imshow(plotly_fig, facet_col=1, animation_frame=0, binary_string=True)
        fig.update_layout(height=600, width=600*plotly_fig.shape[1], title_text="Results")
        with open(save_path, 'a') as f:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    return True

def save_figs2html(plotly_figs_list, save_path):
    fig = px.imshow(plotly_figs.transpose((1,0)), facet_col=1, animation_frame=0, binary_string=True)
    for plotly_figs in plotly_figs_list[1:]:
        fig = px.imshow(plotly_figs.transpose((1,0)), facet_col=1, animation_frame=0, binary_string=True)