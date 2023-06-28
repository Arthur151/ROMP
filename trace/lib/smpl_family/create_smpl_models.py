from smpl_family.smpl import SMPL
from smpl_family.smpla import SMPLA_parser
from smpl_family.smplx import SMPLX
from config import args

def create_model(model_type, model_path=None, **kwargs):
    if model_type == 'smpl':
        model_path = args().smpl_model_path if model_path is None else model_path
        return SMPL(model_path, model_type='smpl', **kwargs)
    if model_type == 'smpla':
        return SMPLA_parser(args().smpla_model_path, args().smil_model_path, baby_thresh=args().baby_threshold, **kwargs)
    if model_type == 'smplx':
        model_path = os.path.join(args().smplx_model_folder, 'SMPLX_NEUTRAL.pth') if model_path is None else model_path
        return SMPLX(model_path, **kwargs)