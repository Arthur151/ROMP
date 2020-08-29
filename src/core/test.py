
from base import *

class Demo(Base):
    def __init__(self):
        super(Demo, self).__init__()
        self.set_up_smplx()
        self._build_model()
        self.save_mesh = False
        self.save_centermap = False
        self.demo_dir = os.path.join(config.project_dir, 'demo')
        print('Initialization finished!')

    def run(self, image_folder):
        vis_size = [1024,1024,3]#[1920,1080]
        self.generator.eval()
        loader_val = self._create_single_data_loader(dataset='internet',train_flag=False, image_folder=image_folder)
        test_save_dir = image_folder+'_results'
        os.makedirs(test_save_dir,exist_ok=True)
        self.visualizer = Visualizer(model_type=self.model_type,resolution =vis_size, input_size=self.input_size, result_img_dir = test_save_dir,with_renderer=True)

        with torch.no_grad():
            for test_iter,data_3d in enumerate(loader_val):
                outputs, centermaps, heatmap_AEs, data_3d_new, reorganize_idx = self.net_forward(data_3d,self.generator,mode='test')
                if not self.save_centermap:
                    centermaps = None
                vis_eval_results = self.visualizer.visulize_result_onorg(outputs['verts'], outputs['verts_camed'], data_3d_new, reorganize_idx, centermaps=centermaps,save_img=True)#

                if self.save_mesh:
                    vids_org = np.unique(reorganize_idx)
                    for idx, vid in enumerate(vids_org):
                        verts_vids = np.where(reorganize_idx==vid)[0]
                        img_path = data_3d['imgpath'][verts_vids[0]]
                        obj_name = (test_save_dir+'/{}'.format(os.path.basename(img_path))).replace('.jpg','.obj').replace('.png','.obj')
                        for subject_idx, batch_idx in enumerate(verts_vids):
                            save_obj(outputs['verts'][batch_idx].detach().cpu().numpy().astype(np.float16), self.smplx.faces_tensor.detach().cpu().numpy(),obj_name.replace('.obj', '_{}.obj'.format(subject_idx)))
                if test_iter%50==0:
                    print(test_iter,'/',len(loader_val))

def main():
    demo = Demo()
    # run the code on demo images
    demo_image_folder = args.demo_image_folder
    if not os.path.exists(demo_image_folder):
        demo_image_folder = os.path.join(demo.demo_dir,'images')
    demo.run(demo_image_folder)


if __name__ == '__main__':
    main()