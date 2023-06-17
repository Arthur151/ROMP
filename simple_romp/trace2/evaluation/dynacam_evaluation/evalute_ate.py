
# evaluation lib: evo
# https://github.com/MichaelGrupp/evo
# pip install evo --upgrade --no-binary evo
import os
try:
    import evo
except:
    os.system("pip install evo --upgrade --no-binary evo")
    import evo

from evo.core.trajectory import PoseTrajectory3D
from evo.core import sync
import evo.main_ape as main_ape
from evo.core.metrics import PoseRelation
from evo.tools import plot
from copy import deepcopy
import matplotlib.pyplot as plt


def make_traj(args) -> PoseTrajectory3D:
    if isinstance(args, tuple):
        traj, tstamps = args
        return PoseTrajectory3D(positions_xyz=traj[:,:3], orientations_quat_wxyz=traj[:,3:], timestamps=tstamps)
    assert isinstance(args, PoseTrajectory3D), type(args)
    return deepcopy(args)

def best_plotmode(traj):
    _, i1, i2 = np.argsort(np.var(traj.positions_xyz, axis=0))
    plot_axes = "xyz"[i2] + "xyz"[i1]
    return getattr(plot.PlotMode, plot_axes)

def plot_trajectory(pred_traj, gt_traj=None, title="", filename="", align=True, correct_scale=True):
    pred_traj = make_traj(pred_traj)

    if gt_traj is not None:
        gt_traj = make_traj(gt_traj)
        gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

        if align:
            pred_traj.align(gt_traj, correct_scale=correct_scale)

    plot_collection = plot.PlotCollection("PlotCol")
    fig = plt.figure(figsize=(8, 8))
    plot_mode = best_plotmode(gt_traj if (gt_traj is not None) else pred_traj)
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title(title)
    if gt_traj is not None:
        plot.traj(ax, plot_mode, gt_traj, '--', 'gray', "Ground Truth")
    plot.traj(ax, plot_mode, pred_traj, '-', 'blue', "Predicted")
    plot_collection.add_figure("traj (error)", fig)
    plot_collection.export(filename, confirm_overwrite=False)
    plt.close(fig=fig)

def evaluate_ate(traj_est, traj_ref, timestamps, seq_name, show_results=False, align=True, vis_folder="trajectory_plots"):
    traj_est = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3],
        orientations_quat_wxyz=traj_est[:,3:],
        timestamps=timestamps)

    traj_ref = PoseTrajectory3D(
        positions_xyz=traj_ref[:,:3],
        orientations_quat_wxyz=traj_ref[:,3:],
        timestamps=timestamps)

    # ATE: considering the translation error only, unit m.
    result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=align, correct_scale=align) # rotation_part
    ate_score = result.stats["rmse"]

    # APE: considering both translation and rotation error only, unit-less.
    full_result = main_ape.ape(traj_ref, traj_est, est_name='grot', 
        pose_relation=PoseRelation.full_transformation, align=align, correct_scale=align)
    ape_score = full_result.stats["rmse"]

    if show_results:
        os.makedirs(vis_folder, exist_ok=True)
        plot_trajectory(traj_est, traj_ref, f"{seq_name} (ATE: {ate_score:.03f})",
                        f"{vis_folder}/{seq_name}.pdf", align=align, correct_scale=align)
    return ate_score, ape_score