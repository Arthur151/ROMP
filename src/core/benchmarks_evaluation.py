
from base import *
from eval import val_result,print_results

class Evaluation(Base):
    def __init__(self):
        super(Evaluation, self).__init__()
        self._build_model_()
        self.model.eval()
        self.test_cfg = {'mode':'train', 'calc_loss': False}
        self.eval_dataset = args.eval_dataset
        self.save_mesh = False
        print('Initialization finished!')

    def run(self):
        if self.eval_dataset == 'pw3d_test':
            data_loader = self._create_single_data_loader(dataset='pw3d', train_flag=False, mode='vibe', split='test',joint_format='lsp14')
        
        MPJPE, PA_MPJPE, eval_results = val_result(self,loader_val=data_loader, evaluation =True)


def main():
    evaluation = Evaluation()
    evaluation.run()

if __name__ == '__main__':
    main()


