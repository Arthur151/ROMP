import matplotlib.pyplot as plt
from pylab import cm
import cv2
import numpy as np

def prepare_wdh_map(d, h, w):
    d_map = np.zeros((d,h,w))
    for ind in range(d):
        d_map[ind] = ind
    h_map = np.zeros((d,h,w))
    for ind in range(h):
        h_map[:,ind] = ind
    w_map = np.zeros((d,h,w))
    for ind in range(w):
        w_map[:,:,ind] = ind

    return [w_map, d_map, h_map]

wdh_map = prepare_wdh_map(64,128,128)

def set_time2close(fig,second):
    timer = fig.canvas.new_timer(interval=second*1000)
    timer.add_callback(plt.close)
    timer.start()

def plot3DHeatmap(heatmap, image, hm2D_fv=None, mm2D_fv=None, motion3D_lines=None, size = 12):
    plt.rcParams['figure.figsize'] = (24, 12)

    d, h, w = heatmap.shape
    if not (type(heatmap) is np.ndarray):
        heatmap = heatmap.detach().cpu().numpy()
        image = image.detach().cpu().numpy().astype(np.uint8)

    fig = plt.figure()
    ax = fig.add_subplot(221, projection='3d')
    set_time2close(fig,2)
    #wdh_map = prepare_wdh_map(d, h, w)
    
    color_map = plt.get_cmap('coolwarm')
    
    cs = heatmap.reshape(-1)
    valid_mask = cs > 0.04
    xs = wdh_map[0].reshape(-1)[valid_mask] * size
    ys = wdh_map[1].reshape(-1)[valid_mask] * size
    zs = wdh_map[2].reshape(-1)[valid_mask] * size
    # to make the visual results look normal
    zs = (h-1) * size - zs

    cs = cs[valid_mask] * size
    scatter_plot = ax.scatter3D(xs, ys, zs, c=cs, cmap=color_map, s=1,alpha=0.5)
    
    ax.set_xlabel('W')
    ax.set_ylabel('D')
    ax.set_zlabel('H')
    ax.set_xlim(xmin=0, xmax=w*size)
    ax.set_ylim(ymin=0, ymax=d*size)
    ax.set_zlim(zmin=0, zmax=h*size)

    ax = fig.add_subplot(222)
    plt.imshow(cv2.addWeighted(cv2.resize(hm2D_fv[:,:,[2,1,0]], (512,512)), 0.5, image[:,:,[2,1,0]], 0.5, 0))

    ax = fig.add_subplot(223, projection='3d')
    draw_motion3D_map(motion3D_lines, ax)

    ax = fig.add_subplot(224)
    plt.imshow(mm2D_fv[:,:,[2,1,0]])

    plt.show()

def draw_motion3D_map(motion3D_lines, ax):
    N = motion3D_lines.shape[1]
    for i in range(N):
        ax.plot(motion3D_lines[:,i,2], motion3D_lines[:,i,0], motion3D_lines[:,i,1]) #, linestyle='dotted'
    
    ax.set_xlabel('W')
    ax.set_ylabel('D')
    ax.set_zlabel('H')
    ax.set_xlim(xmin=0, xmax=512)
    ax.set_ylim(ymin=0, ymax=512)
    ax.set_zlim(zmin=0, zmax=512)


def convert_heatmap(hmap):
    return cv2.applyColorMap((hmap.detach().cpu().numpy() * 255).astype(np.uint8), colormap=cv2.COLORMAP_JET)

def convert_motionmap2motionline(motion_map, heatmap, size=512):
    """
    convert the motion offsets in estimated motion map to the format of 2D motion vector in image:
    the 2D motion vector on image in 2-dim has the 2D coordinates of the start point and the end point. 
    """
    motion_map = motion_map.detach().cpu().numpy()
    heatmap = heatmap.detach().cpu().numpy()
    dim, h, w = motion_map.shape

    hw_map = np.zeros((2,h,w))
    for ind in range(h):
        hw_map[0,ind] = ind
    for ind in range(w):
        hw_map[1,:,ind] = ind
    hw_map = hw_map * size / max(h, w)
    
    valid_mask = heatmap.reshape(-1) > 0.3

    end_coords = hw_map.reshape(2, -1).transpose(1,0)
    start_coords = end_coords - (motion_map[[1,2]] * size / 2).reshape(2, -1).transpose(1,0) 
    motion_line = np.concatenate([start_coords[:,None,[1,0]], end_coords[:,None,[1,0]]], 1) # , [1,0] convert (y,x) to (x,y)
    motion_line = motion_line[valid_mask]
    return motion_line

def convert_motionmap3D2motionline(motion_map, heatmap, size=512):
    """
    convert the motion offsets in estimated motion map to the format of 3D motion vector in image:
    the 3D motion vector on image in 3-dim has the 3D coordinates of the start point and the end point. 
    """
    motion_map = motion_map.detach().cpu().numpy()
    heatmap = heatmap.detach().cpu().numpy()
    dim, d, h, w = motion_map.shape
    rescale = size / max(h, w)

    dhw_map = np.zeros((3,d,h,w))
    for ind in range(d):
        dhw_map[0,ind] = ind
    for ind in range(h):
        dhw_map[1,:,ind] = ind
    for ind in range(w):
        dhw_map[2,:,:,ind] = ind
    # to be consistent with the left-top image origin
    dhw_map[1] = h-1 - dhw_map[1]
    dhw_map = dhw_map * rescale
    
    valid_mask = heatmap > 0.3

    end_coords = dhw_map[:,:,valid_mask].reshape(3, -1).transpose(1,0)
    start_coords = end_coords - (motion_map[:,:,valid_mask] * max(h, w) / 2).reshape(3, -1).transpose(1,0) 
    motion_line = np.stack([start_coords, end_coords])
    return motion_line

def draw_traj_on_image_py(rgb, traj, S=50, linewidth=1, show_dots=False, cmap='coolwarm', maxdist=None):
    # all inputs are numpy tensors
    # rgb is 3 x H x W
    # traj is S x 2
    H, W, C = rgb.shape
    assert(C==3)

    rgb = rgb.astype(np.uint8).copy()

    S1, D = traj.shape
    assert(D==2)

    color_map = cm.get_cmap(cmap)
    S1, D = traj.shape

    for s in range(S1-1):
        if maxdist is not None:
            val = (np.sqrt(np.sum((traj[s]-traj[0])**2))/maxdist).clip(0,1)
            color = np.array(color_map(val)[:3]) * 255 # rgb
        else:
            color = np.array(color_map((s)/max(1,float(S-2)))[:3]) * 255 # rgb

        cv2.line(rgb,
                    (int(traj[s,0]), int(traj[s,1])),
                    (int(traj[s+1,0]), int(traj[s+1,1])),
                    color,
                    linewidth,
                    cv2.LINE_AA)
        if show_dots:
            cv2.circle(rgb, (traj[s,0], traj[s,1]), linewidth*2, color, -1)

    if maxdist is not None:
        val = (np.sqrt(np.sum((traj[-1]-traj[0])**2))/maxdist).clip(0,1)
        color = np.array(color_map(val)[:3]) * 255 # rgb
    else:
        # draw the endpoint of traj, using the next color (which may be the last color)
        color = np.array(color_map((S1-1)/max(1,float(S-2)))[:3]) * 255 # rgb
        
    # color = np.array(color_map(1.0)[:3]) * 255
    cv2.circle(rgb, (traj[-1,0], traj[-1,1]), linewidth*2, color, -1)

    return rgb


def summ_traj2ds_on_rgbs(name, trajs, rgb, frame_ids=None, only_return=False, show_dots=True, cmap='coolwarm', linewidth=1):
    N = len(trajs)
    for i in range(N):
        if cmap=='onediff' and i==0:
            cmap_ = 'spring'
        elif cmap=='onediff':
            cmap_ = 'winter'
        else:
            cmap_ = cmap
        traj = trajs[i].astype(np.int32) # S, 2
        rgb = draw_traj_on_image_py(rgb, traj, S=2, show_dots=show_dots, cmap=cmap_, linewidth=linewidth)
    return rgb


def flow2img(flow_data):
	"""
	convert optical flow into color image
	:param flow_data:
	:return: color image
	"""
	u = flow_data[:, :, 0]
	v = flow_data[:, :, 1]

	UNKNOW_FLOW_THRESHOLD = 1e7
	pr1 = abs(u) > UNKNOW_FLOW_THRESHOLD
	pr2 = abs(v) > UNKNOW_FLOW_THRESHOLD
	idx_unknown = (pr1 | pr2)
	u[idx_unknown] = v[idx_unknown] = 0

	# get max value in each direction
	maxu = -999.
	maxv = -999.
	minu = 999.
	minv = 999.
	maxu = max(maxu, np.max(u))
	maxv = max(maxv, np.max(v))
	minu = min(minu, np.min(u))
	minv = min(minv, np.min(v))

	rad = np.sqrt(u ** 2 + v ** 2)
	maxrad = max(-1, np.max(rad))
	u = u / maxrad + np.finfo(float).eps
	v = v / maxrad + np.finfo(float).eps

	img = compute_color(u, v)

	idx = np.repeat(idx_unknown[:, :, np.newaxis], 3, axis=2)
	img[idx] = 0

	return np.uint8(img)


def compute_color(u, v):
	"""
	compute optical flow color map
	:param u: horizontal optical flow
	:param v: vertical optical flow
	:return:
	"""

	height, width = u.shape
	img = np.zeros((height, width, 3))

	NAN_idx = np.isnan(u) | np.isnan(v)
	u[NAN_idx] = v[NAN_idx] = 0

	colorwheel = make_color_wheel()
	ncols = np.size(colorwheel, 0)

	rad = np.sqrt(u ** 2 + v ** 2)

	a = np.arctan2(-v, -u) / np.pi

	fk = (a + 1) / 2 * (ncols - 1) + 1

	k0 = np.floor(fk).astype(int)

	k1 = k0 + 1
	k1[k1 == ncols + 1] = 1
	f = fk - k0

	for i in range(0, np.size(colorwheel, 1)):
		tmp = colorwheel[:, i]
		col0 = tmp[k0 - 1] / 255
		col1 = tmp[k1 - 1] / 255
		col = (1 - f) * col0 + f * col1

		idx = rad <= 1
		col[idx] = 1 - rad[idx] * (1 - col[idx])
		notidx = np.logical_not(idx)

		col[notidx] *= 0.75
		img[:, :, i] = np.uint8(np.floor(255 * col * (1 - NAN_idx)))

	return img


def make_color_wheel():
	"""
	Generate color wheel according Middlebury color code
	:return: Color wheel
	"""
	RY = 15
	YG = 6
	GC = 4
	CB = 11
	BM = 13
	MR = 6

	ncols = RY + YG + GC + CB + BM + MR

	colorwheel = np.zeros([ncols, 3])

	col = 0

	# RY
	colorwheel[0:RY, 0] = 255
	colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
	col += RY

	# YG
	colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
	colorwheel[col:col + YG, 1] = 255
	col += YG

	# GC
	colorwheel[col:col + GC, 1] = 255
	colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
	col += GC

	# CB
	colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
	colorwheel[col:col + CB, 2] = 255
	col += CB

	# BM
	colorwheel[col:col + BM, 2] = 255
	colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
	col += + BM

	# MR
	colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
	colorwheel[col:col + MR, 0] = 255

	return colorwheel