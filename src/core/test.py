from base import *

class Demo(Base):
    def __init__(self):
        super(Demo, self).__init__()
        self._build_model_()
        self._prepare_modules_()

    def _prepare_modules_(self):
        self.model.eval()
        self.demo_dir = os.path.join(config.project_dir, 'demo')
        self.vis_size = [1024,1024,3]#[1920,1080]
        if not args.webcam and '-1' not in self.gpu:
            self.visualizer = Visualizer(resolution=self.vis_size, input_size=self.input_size,with_renderer=True)
        else:
            self.save_visualization_on_img = False
        if self.save_mesh:
            self.smpl_faces = pickle.load(open(os.path.join(args.smpl_model_path,'smpl','SMPL_NEUTRAL.pkl'),'rb'), encoding='latin1')['f']
        print('Initialization finished!')

    def run(self, image_folder):
        print('Processing {}'.format(image_folder))
        test_save_dir = self.output_dir
        os.makedirs(test_save_dir,exist_ok=True)
        if '-1' not in self.gpu:
            self.visualizer.result_img_dir = test_save_dir
        counter = Time_counter(thresh=1)
            
        internet_loader = self._create_single_data_loader(dataset='internet',train_flag=False, image_folder=image_folder)
        counter.start()
        with torch.no_grad():
            for test_iter,meta_data in enumerate(internet_loader):
                outputs = self.net_forward(meta_data, cfg=self.demo_cfg)
                reorganize_idx = outputs['reorganize_idx'].cpu().numpy()
                counter.count()
                
                if self.save_dict_results:
                    self.reorganize_results(outputs, outputs['meta_data']['imgpath'], reorganize_idx, test_save_dir)
                if self.save_visualization_on_img:
                    vis_eval_results = self.visualizer.visulize_result_onorg(outputs['verts'], outputs['verts_camed'], outputs['meta_data'], \
                    reorganize_idx, centermaps= outputs['center_map']if self.save_centermap else None,save_img=True)#

                if self.save_mesh:
                    vids_org = np.unique(reorganize_idx)
                    for idx, vid in enumerate(vids_org):
                        verts_vids = np.where(reorganize_idx==vid)[0]
                        img_path = outputs['meta_data']['imgpath'][verts_vids[0]]
                        obj_name = (test_save_dir+'/{}'.format(os.path.basename(img_path))).replace('.jpg','.obj').replace('.png','.obj')
                        for subject_idx, batch_idx in enumerate(verts_vids):
                            save_obj(outputs['verts'][batch_idx].detach().cpu().numpy().astype(np.float16), \
                                self.smpl_faces,obj_name.replace('.obj', '_{}.obj'.format(subject_idx)))
                
                if test_iter%50==0:
                    print(test_iter,'/',len(internet_loader))
                counter.start()   

    def reorganize_results(self, outputs, img_paths, reorganize_idx, test_save_dir=None):
        results = {}
        cam_results = outputs['params']['cam'].detach().cpu().numpy().astype(np.float16)
        smpl_pose_results = torch.cat([outputs['params']['global_orient'], outputs['params']['body_pose']],1).detach().cpu().numpy().astype(np.float16)
        smpl_shape_results = outputs['params']['betas'].detach().cpu().numpy().astype(np.float16)
        joints_54 = outputs['j3d'].detach().cpu().numpy().astype(np.float16)
        kp3d_smpl24_results = outputs['joints_smpl24'].detach().cpu().numpy().astype(np.float16)
        kp3d_spin24_results = joints_54[:,constants.joint_mapping(constants.SMPL_ALL_54, constants.SPIN_24)]
        kp3d_op25_results = joints_54[:,constants.joint_mapping(constants.SMPL_ALL_54, constants.OpenPose_25)]
        verts_results = outputs['verts'].detach().cpu().numpy().astype(np.float16)
        pj2d_results = outputs['pj2d'].detach().cpu().numpy().astype(np.float16)

        vids_org = np.unique(reorganize_idx)
        for idx, vid in enumerate(vids_org):
            verts_vids = np.where(reorganize_idx==vid)[0]
            img_path = img_paths[verts_vids[0]]
            results[img_path] = [{} for idx in range(len(verts_vids))]
            for subject_idx, batch_idx in enumerate(verts_vids):
                results[img_path][subject_idx]['cam'] = cam_results[batch_idx]
                results[img_path][subject_idx]['pose'] = smpl_pose_results[batch_idx]
                results[img_path][subject_idx]['betas'] = smpl_shape_results[batch_idx]
                results[img_path][subject_idx]['j3d_smpl24'] = kp3d_smpl24_results[batch_idx]
                results[img_path][subject_idx]['j3d_spin24'] = kp3d_spin24_results[batch_idx]
                results[img_path][subject_idx]['j3d_op25'] = kp3d_op25_results[batch_idx]
                results[img_path][subject_idx]['verts'] = verts_results[batch_idx]
                results[img_path][subject_idx]['pj2d'] = pj2d_results[batch_idx]
                results[img_path][subject_idx]['trans'] = convert_cam_to_3d_trans(cam_results[batch_idx])

        if test_save_dir is not None:
            for img_path, result_dict in results.items():
                name = (test_save_dir+'/{}'.format(os.path.basename(img_path))).replace('.jpg','.npz').replace('.png','.npz')
                # get the results: np.load('/path/to/person_overlap.npz',allow_pickle=True)['results'][()]
                np.savez(name, results=result_dict)
        return results

    def single_image_forward(self,image):
        meta_data = img_preprocess(image, '0', input_size=args.input_size, single_img_input=True)
        if '-1' not in self.gpu:
            meta_data['image'] = meta_data['image'].cuda()
        outputs = self.net_forward(meta_data, cfg=self.demo_cfg)
        return outputs

    def process_video(self, video_file_path=None):
        import keyboard
        from utils.demo_utils import OpenCVCapture, frames2video
        capture = OpenCVCapture(video_file_path)
        video_length = int(capture.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        results, result_frames = {}, []
        for frame_id in range(video_length):
            print('Processing video {}/{}'.format(frame_id, video_length))
            frame = capture.read()
            with torch.no_grad():
                outputs = self.single_image_forward(frame)
            vis_dict = {'image_org': outputs['meta_data']['image_org'].cpu()}
            single_batch_results = self.reorganize_results(outputs,[os.path.basename(video_file_path)+'_'+str(frame_id)],outputs['reorganize_idx'].cpu().numpy())
            results.update(single_batch_results)
            vis_eval_results = self.visualizer.visulize_result_onorg(outputs['verts'], outputs['verts_camed'], vis_dict, reorganize_idx=outputs['reorganize_idx'].cpu().numpy())
            result_frames.append(vis_eval_results[0])
        
        if self.save_dict_results:
            print('Saving parameter results to {}'.format(video_file_path.replace('.mp4', '_results.npz')))
            np.savez(video_file_path.replace('.mp4', '_results.npz'), results=results)

        video_save_name = video_file_path.replace('.mp4', '_results.mp4')
        print('Writing results to {}'.format(video_save_name))
        frames2video(result_frames, video_save_name, fps=args.fps_save)
            
    def webcam_run_local(self, video_file_path=None):
        '''
        20.9 FPS of forward prop. on 1070Ti
        '''
        print('run on local')
        import keyboard
        from utils.demo_utils import OpenCVCapture, Image_Reader 
        if 'tex' in args.webcam_mesh_color:
            from utils.demo_utils import vedo_visualizer as Visualizer
        else:
            from utils.demo_utils import Open3d_visualizer as Visualizer
        capture = OpenCVCapture(video_file_path)
        visualizer = Visualizer()
        print('Initialization is down')

        # Warm-up
        for i in range(10):
            self.single_image_forward(np.zeros((512,512,3)).astype(np.uint8))
        counter = Time_counter(thresh=1)
        while True:
            start_time_perframe = time.time()
            frame = capture.read()
            if frame is None:
                continue
            
            counter.start()
            with torch.no_grad():
                outputs = self.single_image_forward(frame)
            counter.count()
            counter.fps()

            if outputs is not None and outputs['detection_flag']:
                if args.show_single:
                    verts = outputs['verts'].cpu().numpy()
                    verts = verts * 50 + np.array([0, 0, 100])
                    break_flag = visualizer.run(verts[0],frame)
                else:
                    verts = outputs['verts_camed'].cpu().numpy()
                    verts = verts * 50 + np.array([0, 0, 100])
                    break_flag = visualizer.run_multiperson(verts,frame)
                if break_flag:
                    break

    def webcam_run_remote(self):
        print('run on remote')
        from utils.remote_server_utils import Server_port_receiver
        capture = Server_port_receiver()

        while True:
            frame = capture.receive()
            if isinstance(frame,list):
                continue
            with torch.no_grad():
                outputs = self.single_image_forward(frame)
            if outputs is not None:
                verts = outputs['verts'][0].cpu().numpy()
                verts = verts * 50 + np.array([0, 0, 100])
                capture.send(verts)
            else:
                capture.send(['failed'])

class Time_counter():
    def __init__(self,thresh=0.1):
        self.thresh=thresh
        self.runtime = 0
        self.frame_num = 0

    def start(self):
        self.start_time = time.time()

    def count(self):
        time_cost = time.time()-self.start_time
        if time_cost<self.thresh:
            self.runtime+=time_cost
            self.frame_num+=1
        self.start()

    def fps(self):
        print('average per-frame runtime:',self.runtime/self.frame_num)
        print('FPS: {}, not including visualization time. '.format(self.frame_num/self.runtime))

    def reset(self):
        self.runtime = 0
        self.frame_num = 0

def main():
    demo = Demo()
    if args.webcam:
        print('Running the code on webcam demo')
        if args.run_on_remote_server:
            demo.webcam_run_remote()
        else:
            demo.webcam_run_local()
    elif args.video_or_frame:
        print('Running the code on video ',args.input_video_path)
        demo.process_video(args.input_video_path)
    else:
        demo_image_folder = args.demo_image_folder
        if not os.path.exists(demo_image_folder):
            print('Running the code on the demo images')
            demo_image_folder = os.path.join(demo.demo_dir,'images')
        demo.run(demo_image_folder)


if __name__ == '__main__':

    main()