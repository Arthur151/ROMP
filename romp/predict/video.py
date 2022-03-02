import sys 
whether_set_yml = ['configs_yml' in input_arg for input_arg in sys.argv]
if sum(whether_set_yml)==0:
    default_webcam_configs_yml = "--configs_yml=configs/video.yml"
    print('No configs_yml is set, set it to the default {}'.format(default_webcam_configs_yml))
    sys.argv.append(default_webcam_configs_yml)
from .image import *
import keyboard
from utils.demo_utils import frames2video, video2frame
#from tracking.tracker import Tracker
from norfair import Detection, Tracker, Video, draw_tracked_objects
import norfair
from utils.temporal_optimization import create_OneEuroFilter, temporal_optimize_result
import itertools

class Video_processor(Image_processor):
    def __init__(self, **kwargs):
        super(Video_processor, self).__init__(**kwargs)

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
        if self.save_visualization_on_img:
            self.visualizer.result_img_dir = self.output_dir 
        counter = Time_counter(thresh=1)

        internet_loader = self._create_single_data_loader(dataset='internet', train_flag=False, shuffle=False, file_list=frame_list)
        counter.start()

        results_frames = {}
        save_frame_list = []
        results_track_video = {}
        video_track_ids = {}
        subjects_motion_sequences = {}

        if self.make_tracking:
            #tracker = Tracker()
            tracker = Tracker(distance_function=euclidean_distance, distance_threshold=30)
            
        if self.temporal_optimization:
            filter_dict = {}
        
        if self.show_mesh_stand_on_image:
            from visualization.vedo_visualizer import Vedo_visualizer
            visualizer = Vedo_visualizer()
            stand_on_imgs_frames = []

        for test_iter,meta_data in enumerate(internet_loader):
            outputs = self.net_forward(meta_data, cfg=self.demo_cfg)
            reorganize_idx = outputs['reorganize_idx'].cpu().numpy()
            counter.count(self.val_batch_size)
            results = self.reorganize_results(outputs, outputs['meta_data']['imgpath'], reorganize_idx)

            img_paths = [img_path for img_path in results]
            results_track = {os.path.splitext(os.path.basename(img_path))[0]:results[img_path] for img_path in img_paths}
            for frame_id in sorted(list(results_track.keys())):
                params_dict_new = {'cam':[], 'betas':[], 'poses':[]}
                item_names = list(params_dict_new.keys())
                reorganize_idx_uq = np.unique(reorganize_idx)
                to_org_inds = np.array([np.where(reorganize_idx==ind)[0][0] for ind in reorganize_idx_uq])

                if self.show_largest_person_only:
                    max_id = np.argmax(np.array([result['cam'][0] for result in results_track[frame_id]]))
                    results_track[frame_id] = [results_track[frame_id][max_id]]
                    video_track_ids[frame_id] = [0]
                elif self.make_tracking:
                    detections = [Detection(points=(result['cam'][[1,2]]+1)/2.*args().input_size) for result in results_track[frame_id]]
                    if test_iter==0:
                        for _ in range(8):
                            tracked_objects = tracker.update(detections=detections)
                    tracked_objects = tracker.update(detections=detections)
                    tracked_ids = get_tracked_ids(detections, tracked_objects)
                    # frame = np.ones([512,512,3])
                    # norfair.draw_tracked_objects(frame, tracked_objects,id_size=5,id_thickness=2)
                    # cv2.imshow('tracking results', frame[:,:,::-1])
                    # cv2.waitKey(1)
                    video_track_ids[frame_id] = tracked_ids

            if self.temporal_optimization or self.show_largest_person_only:
                reorganize_idx_new, img_paths_new = [], []
                for fid, frame_id in enumerate(sorted(list(results_track.keys()))):
                    for sid, tid in enumerate(video_track_ids[frame_id]):
                        if self.temporal_optimization:
                            if tid not in filter_dict:
                                filter_dict[tid] = create_OneEuroFilter(args().smooth_coeff)
                            results_track[frame_id][sid] = temporal_optimize_result(results_track[frame_id][sid], filter_dict[tid])
                        
                        for item in item_names:
                            params_dict_new[item].append(torch.from_numpy(results_track[frame_id][sid][item]))
                        reorganize_idx_new.append(fid)
                        img_paths_new.append(img_paths[fid])
                        if tid not in subjects_motion_sequences:
                            subjects_motion_sequences[tid] = {}
                        subjects_motion_sequences[tid][frame_id] = results_track[frame_id][sid]

                # update the vertices
                for item in item_names:
                    params_dict_new[item] = torch.stack(params_dict_new[item]).cuda()
                outputs['meta_data']['offsets'] = outputs['meta_data']['offsets'][to_org_inds][reorganize_idx_new]
                with autocast():
                    outputs.update(self.model.module._result_parser.params_map_parser.recalc_outputs(params_dict_new, outputs['meta_data']))
                
                outputs['meta_data']['image'] = outputs['meta_data']['image'][to_org_inds][reorganize_idx_new]
                results = self.reorganize_results(outputs, img_paths_new, reorganize_idx_new)
                outputs['reorganize_idx'] = torch.Tensor(reorganize_idx_new)
                outputs['meta_data']['imgpath'] = img_paths_new

            if self.save_dict_results:
                save_result_dict_tonpz(results, self.output_dir)
                
            if self.show_mesh_stand_on_image:
                poses = []
                for frame_id in sorted(list(results_track.keys())):
                    for result in results_track[frame_id]:
                        poses.append(result['poses'])
                pose = np.array(poses)
                verts = self.character_model(pose)['verts'] if self.character == 'nvxia' else outputs['verts']
                rotate_frames = [0] if self.surrounding_camera else []
                stand_on_imgs = visualizer.plot_multi_meshes_batch(verts, outputs['params']['cam'], outputs['meta_data'], \
                    outputs['reorganize_idx'].cpu().numpy(), interactive_show=self.interactive_vis, rotate_frames=rotate_frames)
                stand_on_imgs_frames += stand_on_imgs

            if self.save_visualization_on_img:
                show_items_list = ['org_img', 'mesh']
                if self.save_centermap:
                    show_items_list.append('centermap')
                results_dict, img_names = self.visualizer.visulize_result(outputs, outputs['meta_data'], \
                    show_items=show_items_list, vis_cfg={'settings':['put_org']}, save2html=False)

                if 'mesh_rendering_orgimgs' in results_dict:
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

        if len(subjects_motion_sequences)>0:
            save_dict_path = os.path.join(self.output_dir, video_basename+'_ts_results.npz')
            print('Saving parameter results to {}'.format(save_dict_path))
            np.savez(save_dict_path, results=subjects_motion_sequences)

        if len(save_frame_list)>0 and self.save_visualization_on_img:
            video_save_name = os.path.join(self.output_dir, video_basename+'_results.mp4')
            print('Writing results to {}'.format(video_save_name))
            frames2video(sorted(save_frame_list), video_save_name, fps=self.fps_save)

        if self.show_mesh_stand_on_image and self.save_visualization_on_img:
            video_save_name = os.path.join(self.output_dir, video_basename+'_soi_results.mp4')
            print('Writing results to {}'.format(video_save_name))
            frames2video(stand_on_imgs_frames, video_save_name, fps=self.fps_save)
        return results_frames


def get_tracked_ids(detections, tracked_objects):
    tracked_ids_out = np.array([obj.id for obj in tracked_objects])
    tracked_points = np.array([obj.last_detection.points for obj in tracked_objects])
    org_points = np.array([obj.points for obj in detections])
    tracked_ids = [tracked_ids_out[np.argmin(np.linalg.norm(tracked_points-point[None], axis=1))] for point in org_points]
    return tracked_ids

def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

def main():
    with ConfigContext(parse_args(sys.argv[1:])) as args_set:
        print('Loading the configurations from {}'.format(args_set.configs_yml))
        processor = Video_processor(args_set=args_set)
        print('Processing video: ',args_set.inputs)
        processor.process_video(args_set.inputs)

if __name__ == '__main__':
    main()
