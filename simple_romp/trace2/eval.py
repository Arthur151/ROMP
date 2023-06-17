import os
import numpy as np

from .main import TRACE
from .utils.eval_utils import update_eval_seq_cfgs, get_evaluation_sequence_dict
from .utils.infer_settings import trace_settings
from .utils.utils import preds_save_paths

from .evaluation.eval_dynacam import evaluate_panorama, evaluate_translation
from .evaluation.evaluate_tracking import evaluate_trackers_mupots, evaluate_trackers_dyna3dpw
from .evaluation.eval_3DPW import evaluate_3dpw_results

datasets_dir = {
    'DynaCam-Panorama': '/home/yusun/DataCenter/my_datasets/DynaCam',
    'DynaCam-Translation': '/home/yusun/DataCenter/my_datasets/DynaCam',
    'mupots': '/home/yusun/DataCenter/datasets/MultiPersonTestSet',
    'Dyna3DPW': '/home/yusun/DataCenter/datasets/Dyna3DPW',
    '3DPW': '/home/yusun/DataCenter/datasets/3DPW',}

eval_functions = {
    'DynaCam-Panorama': evaluate_panorama, 'DynaCam-Translation': evaluate_translation, \
    'mupots': evaluate_trackers_mupots, 'Dyna3DPW': evaluate_trackers_dyna3dpw, '3DPW': evaluate_3dpw_results}

class Evaluator(TRACE):
    def __init__(self, args):
        super(Evaluator, self).__init__(args)
    
    def update_sequence_cfs(self, seq_name):
        return update_eval_seq_cfgs(seq_name, self.default_seq_cfgs, ds_name=self.eval_dataset)

    def check_load_previous_results(self, seq_name):
        save_paths = preds_save_paths(self.results_save_dir, prefix=seq_name)
        return os.path.exists(save_paths.seq_results_save_path)

def main():
    args = trace_settings()
    args.results_save_dir += f'-{args.eval_dataset}'
    args.save_video=False
    evaluator = Evaluator(args)

    sequence_dict = get_evaluation_sequence_dict(datasets=args.eval_dataset, dataset_dir=datasets_dir[args.eval_dataset])
    for seq_name, frame_paths in sequence_dict.items():
        if evaluator.check_load_previous_results(os.path.basename(seq_name)): #and os.path.basename(seq_name) not in ['TS16']:
            continue
        outputs, tracking_results, kp3d_results, imgpaths = evaluator({seq_name: frame_paths})
        evaluator.save_results(outputs, tracking_results, kp3d_results, imgpaths)
    
    eval_functions[args.eval_dataset](args.results_save_dir, datasets_dir[args.eval_dataset], vis=False)

if __name__ == '__main__':
    main()