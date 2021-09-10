from .base_predictor import *

class Webcam_processor(Predictor):
    def __init__(self):
        super(Webcam_processor, self).__init__()

    def webcam_run_local(self, video_file_path=None):
        '''
        20.9 FPS of forward prop. on 1070Ti
        '''
        print('run on local')
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
            print(frame.shape)
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
    input_args = sys.argv[1:]
    if sum(['configs_yml' in input_arg for input_arg in input_args])==0:
        input_args.append("--configs_yml=configs/webcam.yml")
    with ConfigContext(parse_args(input_args)):
        processor = Webcam_processor()
        print('Running the code on webcam demo')
        if args().run_on_remote_server:
            processor.webcam_run_remote()
        else:
            processor.webcam_run_local()

if __name__ == '__main__':
    main()