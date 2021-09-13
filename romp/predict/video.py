import sys 
whether_set_yml = ['configs_yml' in input_arg for input_arg in sys.argv]
if sum(whether_set_yml)==0:
    default_webcam_configs_yml = "--configs_yml=configs/video.yml"
    print('No configs_yml is set, set it to the default {}'.format(default_webcam_configs_yml))
    sys.argv.append(default_webcam_configs_yml)
from .image import *
import keyboard
from utils.demo_utils import frames2video, video2frame
import itertools

class Video_processor(Image_processor):
    def __init__(self):
        super(Video_processor, self).__init__()

    @staticmethod
    def toframe(video_file_path):
        assert isinstance(video_file_path, str), \
            print('We expect the input video file path is str, while recieved {}'.format(video_file_path))
        video_basename, video_ext = os.path.splitext(video_file_path)
        assert video_ext in constants.video_exts, \
            print('Video format {} is not currently supported, please convert it to the frames by yourself.'.format(video_ext))
        frame_list = video2frame(video_file_path, frame_save_dir=video_basename+'_frames')
        return video_basename, frame_list

    @torch.no_grad()
    def process_video(self, video_file_path):
        if os.path.isdir(video_file_path):
            frame_list = collect_image_list(image_folder=video_file_path, collect_subdirs=False, img_exts=constants.img_exts)
            frame_list = sorted(frame_list)
            video_basename = video_file_path
        elif os.path.exists(video_file_path):
            video_basename, frame_list = self.toframe(video_file_path)
        else:
            raise('{} not exists!'.format(video_file_path))
        print('Processing {} frames of video {}, saving to {}'.format(len(frame_list), video_basename, self.output_dir))
        video_basename = os.path.basename(video_basename)
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.visualizer.result_img_dir = self.output_dir 
        counter = Time_counter(thresh=1)

        internet_loader = self._create_single_data_loader(dataset='internet', train_flag=False, shuffle=False, file_list=frame_list)
        counter.start()

        results_frames = {}
        save_frame_list = []

        for test_iter,meta_data in enumerate(internet_loader):
            outputs = self.net_forward(meta_data, cfg=self.demo_cfg)
            reorganize_idx = outputs['reorganize_idx'].cpu().numpy()
            counter.count(self.val_batch_size)
            results = self.reorganize_results(outputs, outputs['meta_data']['imgpath'], reorganize_idx)

            if args().show_largest_person_only:
                results_track = {int(os.path.splitext(os.path.basename(img_path))[0]):result for img_path, result in results.items()}
                for frame_id in sorted(list(results_track.keys())):
                    max_id = np.argmax(np.array([result['cam'][0] for result in results_track[frame_id]]))
                    results_track[frame_id] = [results_track[frame_id][max_id]]
                    video_track_ids[frame_id] = [0]

            if self.save_dict_results:
                save_result_dict_tonpz(results, self.output_dir)
                
            if self.save_visualization_on_img:
                show_items_list = ['org_img', 'mesh']
                if self.save_centermap:
                    show_items_list.append('centermap')
                results_dict, img_names = self.visualizer.visulize_result(outputs, outputs['meta_data'], \
                    show_items=show_items_list, vis_cfg={'settings':['put_org']}, save2html=False)

                for img_name, mesh_rendering_orgimg in zip(img_names, results_dict['mesh_rendering_orgimgs']['figs']):
                    save_name = os.path.join(self.output_dir, os.path.basename(img_name))
                    cv2.imwrite(save_name, cv2.cvtColor(mesh_rendering_orgimg, cv2.COLOR_RGB2BGR))
                    save_frame_list.append(save_name)
                del results_dict

            if self.save_mesh:
                save_meshes(reorganize_idx, outputs, self.output_dir, self.smpl_faces)
            
            if test_iter%8==0:
                print('Processed {} / {} frames'.format(test_iter * self.val_batch_size, len(internet_loader.dataset)))
            counter.start()
            results_frames.update(results)

        if self.save_dict_results:
            save_dict_path = os.path.join(self.output_dir, video_basename+'_results.npz')
            print('Saving parameter results to {}'.format(save_dict_path))
            np.savez(save_dict_path, results=results_frames)

        if len(save_frame_list)>0:
            video_save_name = os.path.join(self.output_dir, video_basename+'_results.mp4')
            print('Writing results to {}'.format(video_save_name))
            frames2video(sorted(save_frame_list), video_save_name, fps=args().fps_save)

def main():
    with ConfigContext(parse_args(sys.argv[1:])) as args:
        print('Loading the configurations from {}'.format(args.configs_yml))
        processor = Video_processor()
        print('Processing video: ',args.inputs)
        processor.process_video(args.inputs)

if __name__ == '__main__':
    main()