import torch.multiprocessing as mp
import keyboard
import sys,os 
from PIL import Image
import torchvision
sys.path.append(os.path.abspath(__file__).replace('utils/multiprocess.py',''))
from core.base import *
from core.test import Time_counter
from utils.demo_utils import OpenCVCapture, Open3d_visualizer 

class Multiprocess(Base):
    def __init__(self):
        self.run_single_camera()
        
    def set_up_model_pool(self):
        self.model_pool = []
        for i in range(self.model_number):
            self.model_pool.append(Base())

    def single_image_forward(self,image):
        image_size = image.shape[:2][::-1]
        image_org = Image.fromarray(image)
        
        resized_image_size = (float(self.input_size)/max(image_size) * np.array(image_size) // 2 * 2).astype(np.int)[::-1]
        padding = tuple((self.input_size-resized_image_size)[::-1]//2)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resized_image_size, interpolation=3),
            torchvision.transforms.Pad(padding, fill=0, padding_mode='constant'),
            ])
        image = torch.from_numpy(np.array(transform(image_org))).unsqueeze(0).cuda().contiguous().float()
        outputs, centermaps, heatmap_AEs, _, reorganize_idx = self.net_forward(None,self.generator,image,mode='test')
        return outputs

    def image_put(self, q):
        self.capture = OpenCVCapture()
        time.sleep(3)
        while True:
            if q.qsize() > 2:
                q.get() 
            q.put(self.capture.read())


    def image_get(self, q, q_vis):
        super(Multiprocess, self).__init__()
        self.set_up_smplx()
        self._build_model()
        self.generator.eval()
        for i in range(10):
            self.single_image_forward(np.zeros((512,512,3)).astype(np.uint8))
        while True:
            try:
                frame = q.get()
                with torch.no_grad():
                    outputs = self.single_image_forward(frame)
                q_vis.put((frame,outputs))
            except Exception as error:
                print(error)
                self.endprocess()

    def show_results(self, q):
        '''
        17.5 FPS of entire process on 1080
        '''
        self.visualizer = Open3d_visualizer()
        self.counter = Time_counter(thresh=0.1)
        time.sleep(4)
        start_flag = 1
        while True:
            try:
                if start_flag:
                    self.counter.start()
                frame,outputs = q.get()
                start_flag=0
                break_flag = self.visualize(frame,outputs)
                self.counter.count()
                self.counter.fps()
                if break_flag:
                    self.endprocess()
            except Exception as error:
                print(error)
                #self.endprocess()

    def visualize(self,frame,outputs):
        verts = outputs['verts'][0].cpu().numpy()
        verts = verts * 50 + np.array([0, 0, 100])
        break_flag = self.visualizer.run(verts,frame)
        return break_flag


    def run_single_camera(self):
        queue = mp.Queue(maxsize=3)
        queue_vis = mp.Queue(maxsize=3)
        self.processes = [mp.Process(target=self.image_put, args=(queue,)),
                    mp.Process(target=self.image_get, args=(queue,queue_vis,)),
                    mp.Process(target=self.show_results, args=(queue_vis,))]

        [process.start() for process in self.processes]
        [process.join() for process in self.processes]


    def endprocess(self):
        [process.terminate() for process in self.processes]
        [process.join() for process in self.processes]



def main(vedio_path):
    mulp = Multiprocess()
    mulp.run_single_camera()

if __name__ == '__main__':
    main()
