from .evaluate_tracking import evaluate_trackers
from .eval_kp3ds import eval_kp3ds
from ..utils.eval_utils import adjust_tracking_results

def evaluate_predictions(ds_name, kp3d_results, tracking_results, tracking_matrix_save_path, eval_hard_seq=False):
    eval_results = {}
    if ds_name in ['mupots', 'pw3d', 'cmup']:
        eval_results.update(eval_kp3ds(kp3d_results, dataset=ds_name, eval_hard_seq=eval_hard_seq))
    if ds_name in ['posetrack', 'mupots', 'Dyna3DPW', 'cmup']:
        tracking_results = adjust_tracking_results(tracking_results)
        eval_results.update(evaluate_trackers(tracking_results, tracking_matrix_save_path, dataset=ds_name, eval_hard_seq=eval_hard_seq))
    return eval_results