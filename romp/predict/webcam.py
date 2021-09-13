import sys 
whether_set_yml = ['configs_yml' in input_arg for input_arg in sys.argv]
if sum(whether_set_yml)==0:
    default_webcam_configs_yml = "--configs_yml=configs/webcam.yml"
    print('No configs_yml is set, set it to the default {}'.format(default_webcam_configs_yml))
    sys.argv.append(default_webcam_configs_yml)
from .base_predictor import *

class Webcam_processor(Predictor):
    def __init__(self, **kwargs):
        super(Webcam_processor, self).__init__(**kwargs)

    def webcam_run_local(self, video_file_path=None):
        '''
        24.4 FPS of forward prop. on 1070Ti
        '''
        import keyboard
        from utils.demo_utils import OpenCVCapture, Image_Reader 
        capture = OpenCVCapture(video_file_path, show=True)
        print('Initialization is down')
        frame_id = 0

        from utils.demo_utils import Open3d_visualizer
        visualizer = Open3d_visualizer(multi_mode=False)
        # Warm-up
        for i in range(10):
            self.single_image_forward(np.zeros((512,512,3)).astype(np.uint8))
        counter = Time_counter(thresh=1)
        while True:
            start_time_perframe = time.time()
            frame = capture.read()
            if frame is None:
                continue

            frame_id+=1
            counter.start()
            with torch.no_grad():
                outputs = self.single_image_forward(frame)
            counter.count()
            counter.fps()

            if outputs is not None and outputs['detection_flag']:
                reorganize_idx = outputs['reorganize_idx'].cpu().numpy()
                results = self.reorganize_results(outputs, [frame_id for _ in range(len(reorganize_idx))], reorganize_idx)

                max_id = np.argmax(np.array([result['cam'][0] for result in results[frame_id]]))
                verts_largest = results[frame_id][max_id]['verts'] * 50 + np.array([0, 0, 100])
                visualizer.run(verts_largest)

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

def main():
    with ConfigContext(parse_args(sys.argv[1:])) as args_set:
        print('Loading the configurations from {}'.format(args_set.configs_yml))
        processor = Webcam_processor(args_set=args_set)
        print('Running the code on webcam demo')
        if args_set.run_on_remote_server:
            processor.webcam_run_remote()
        else:
            processor.webcam_run_local()

if __name__ == '__main__':
    main()