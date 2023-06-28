
from base import *
from lib.models.build import build_temporal_model
from lib.datasets.mixed_dataset import MixedDataset, SingleVideoDataset
from lib.utils.video_utils import ordered_organize_frame_outputs_to_clip
from lib.loss_funcs import Loss, Learnable_Loss
from lib import config
np.set_printoptions(precision=2, suppress=True)
from raft.process import FlowExtract

class Trainer(Base):
    def __init__(self):
        super(Trainer, self).__init__()
        self.determine_root_dir()
        self._load_image_model_()
        self._build_video_model_()
        self.flow_estimator = FlowExtract(torch.device('cuda:{}'.format(self.gpus[0] if len(self.gpus)==1 else self.gpus[1])))

        self._build_optimizer_video_()
        self.mutli_task_uncertainty_weighted_loss = Learnable_Loss().to(torch.device(f'cuda:{self.train_devices[0]}'))

        self.loader = self._create_data_loader_video_(train_flag=True)
        
        self.video_train_cfg = {'mode':'matching_gts', 'sequence_input':True, 'is_training':True, 'update_data': True, 'calc_loss': True if self.model_return_loss else False, \
                    'input_type':'sequence', 'with_nms':False, 'with_2d_matching':True, 'new_training': args().new_training, 'regress_params':True, 'traj_conf_threshold':0.12}
        
        self.seq_cacher = {}
        logging.info('Initialization of Trainer finished!')
    
    def determine_root_dir(self):
        local_root_dir = '/home/yusun'
        remote_root_dir = '/home/sunyu15'
        self.show_tracking_results = False
        if os.path.isdir(local_root_dir):
            self.root_dir = local_root_dir
            self.tracking_results_save_dir = '/home/yusun/DataCenter/demo_results/tracking_results'
            self.dataset_dir = '/home/yusun/DataCenter/datasets'
            self.model_path = os.path.join('/home/yusun/Infinity/project_data/trace_data/trained_models', os.path.basename(self.model_path))
            self.temp_model_path = self.temp_model_path #os.path.join('/home/yusun/Infinity/project_data/romp_data/trained_models', os.path.basename(self.temp_model_path))
            self.show_tracking_results = False
        elif os.path.isdir(remote_root_dir):
            self.root_dir = remote_root_dir
            self.tracking_results_save_dir = os.path.join(remote_root_dir, 'tracking_results')
            self.dataset_dir = '/home/sunyu15/datasets'
        else:
            raise NotImplementedError("both path : {} and {} don't exist".format(local_root_dir, remote_root_dir))
    
    def _load_image_model_(self):
        model = build_model(self.backbone, self.model_version, with_loss=False)
        drop_prefix = ''
        model = load_model(self.model_path, model, prefix='module.',drop_prefix=drop_prefix, fix_loaded=True) #
        if not args().train_backbone:
            fix_backbone(model, exclude_key=['backbone.','head.'])

        if self.train_backbone:
            self.image_model = nn.DataParallel(model.cuda())
        else:
            self.image_model_device_id = self.gpus[0] if len(self.gpus)==1 else self.gpus[1] # change to 1,-1 while no evaluation. because putting model on different gpus would cause CUDA error during evaluation.
            torch.cuda.set_device(self.image_model_device_id) # To fix the CUDA error: an illegal memory access was encountered
            self.image_model_device = torch.device(f'cuda:{self.image_model_device_id}')
            self.local_device = torch.device(f'cuda:{self.image_model_device_id}')
            self.image_model = nn.DataParallel(model.to(self.image_model_device), device_ids=[self.image_model_device_id])
        if not args().train_backbone:
            self.image_model = self.image_model.eval()
        else:
            self.image_model = self.image_model.train()
    
    def _build_video_model_(self):
        logging.info('start building learnable video model.')
        temporal_model = build_temporal_model(
            model_type=args().tmodel_type, head=args().tmodel_version)
        if len(self.temp_model_path)>0:
            prefix = 'module.' if 'TROMP_v2' in self.temp_model_path else ''
            #print(torch.load(self.temp_model_path).keys())
            temporal_model = load_model(self.temp_model_path, temporal_model, prefix=prefix, drop_prefix='', fix_loaded=False) #module.
            if self.loading_bev_head_parameters:
                copy_state_dict(temporal_model.state_dict(),  torch.load(self.model_path), prefix = 'module.')
        
        self.train_devices = self.gpus          
        self.temp_model_device = torch.device(f'cuda:{self.train_devices[0]}')
        if self.master_batch_size!=-1:
            self.temporal_model = DataParallel(temporal_model.to(self.temp_model_device), device_ids=self.train_devices, chunk_sizes=self.chunk_sizes) #device_ids=self.gpus, 
        else:
            self.temporal_model = nn.DataParallel(temporal_model.to(self.temp_model_device), device_ids=self.train_devices)

    def _build_optimizer_video_(self):
        if not args().train_backbone:
            self.optimizer = torch.optim.Adam(list(self.image_model.parameters()) + list(self.temporal_model.parameters()), lr = self.lr)
        else:
            self.optimizer = torch.optim.Adam(self.temporal_model.parameters(), lr = self.lr)
        if self.model_precision=='fp16':
            self.scaler = GradScaler()
        self.e_sche = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60,80], gamma = self.adjust_lr_factor)

        logging.info('finished build model.') 
    
    def _create_single_video_sequence_data_loader(self, **kwargs):
        logging.info(
            'gathering a single video datasets, loading a sequence at each time.')
        dataset = SingleVideoDataset(**kwargs)
        batch_size = self.batch_size if kwargs['train_flag'] else self.val_batch_size
        batch_sampler = SequentialBatchSampler(
            'ordered', False, batch_size, dataset)
        data_loader = DataLoader(
            dataset=dataset, batch_sampler=batch_sampler, pin_memory=True, num_workers=self.nw)
        return data_loader
    
    def _create_data_loader_video_(self, train_flag=True):
        dataset_names = self.datasets.split(',')
        loading_modes = ['video_relative' for _ in range(len(dataset_names))]
        datasets = MixedDataset(dataset_names, self.sample_prob_dict, loading_modes=loading_modes, train_flag=train_flag)
        # use the last gpu to learn from images.
        batch_size = self.batch_size
        print('Loading video samples with batch size', batch_size)
        data_loader = DataLoader(dataset = datasets, batch_size = batch_size, shuffle =True,\
                    drop_last = True if train_flag else False, pin_memory = True,num_workers = self.nw)
        return data_loader
    
    def reorganize_meta_data(self, meta_data, sampled_ids):
        new_meta_data = {}
        for key in meta_data:
            try:
                if isinstance(meta_data[key], torch.Tensor):
                    new_meta_data[key] = meta_data[key][sampled_ids]
                elif isinstance(meta_data[key], list):
                    new_meta_data[key] = [meta_data[key][ind]
                                        for ind in sampled_ids]
                else:
                    print('Error in reorganizing the meta data:',
                        key, meta_data[key])
            except:
                print('error!!!', key, len(meta_data[key]), sampled_ids)
            print(key, len(new_meta_data[key]))
        return new_meta_data
    
    def reorganize_clip_data(self, meta_data, cfg_dict):
        """Each batch contains multiple video clips, this function reorganize them (0,1,2,3,4,5,6,7,8,9,10,11,12,13, ...)
        to each 7 small clips [(0,1,2,3,4,5,6), (7,8,9,10,11,12,13), ...] """

        trajectory_info, meta_data['subject_ids'] = ordered_organize_frame_outputs_to_clip(
            meta_data['seq_inds'], person_centers=meta_data['person_centers'], track_ids=meta_data['subject_ids'], \
            cam_params=meta_data['cams'], cam_mask=meta_data['cam_mask'])
        meta_data.update(trajectory_info)
        return meta_data
    
    def network_forward(self, temporal_model, image_model, meta_data, cfg_dict):
        ds_org, imgpath_org = get_remove_keys(meta_data,keys=['data_set','imgpath'])
        with autocast():
            if self.train_backbone:
                image_inputs = {'image':meta_data['image'].cuda()}
            else:
                image_inputs = {'image':meta_data['image'].to(self.local_device)} #, 'batch_ids':torch.arange(len(meta_data['image'])).to(self.local_device)}
            image_outputs = image_model(image_inputs, **{'mode':'extract_img_feature_maps'})

            #sequence_mask = meta_data['seq_inds'][:,3].bool().to(self.local_device)
            meta_data = self.reorganize_clip_data(meta_data, cfg_dict)

            meta_data['batch_ids'] = meta_data['seq_inds'][:,2]
            temp_inputs = {'image_feature_maps': image_outputs['image_feature_maps'].to(self.temp_model_device), 'seq_inds': meta_data['seq_inds'].to(self.temp_model_device)} # , 'image':meta_data['image'].to(self.local_device)            
            if not args().train_backbone:
                temp_inputs['image_feature_maps'] = temp_inputs['image_feature_maps'].detach()
            if args().use_optical_flow:
                optical_flows = self.flow_estimator(image_inputs['image'], meta_data['seq_inds'])
                #print(optical_flows.shape, 'flow max min', torch.max(optical_flows), torch.min(optical_flows), 'feature map max min',torch.max(temp_inputs['image_feature_maps']), torch.min(temp_inputs['image_feature_maps']))
                #show_seq_flow(image_inputs['image'], optical_flows)
                temp_inputs['optical_flows'] = optical_flows.to(self.temp_model_device)
            outputs = temporal_model(temp_inputs, meta_data, **cfg_dict)
            #outputs = self.params_map_parser(outputs, outputs['meta_data'])

        meta_data.update({'imgpath':imgpath_org, 'data_set':ds_org})
        outputs['meta_data']['data_set'], outputs['meta_data']['imgpath'] = reorganize_items([ds_org, imgpath_org], outputs['reorganize_idx'].cpu().numpy())
        return outputs, image_outputs

    def train_step(self, meta_data):
        self.optimizer.zero_grad()
        outputs, BEV_outputs = self.network_forward(
           self.temporal_model, self.image_model, meta_data, self.video_train_cfg)
        if not self.model_return_loss:
           outputs.update(self._calc_loss(outputs))
        loss, outputs = self.mutli_task_uncertainty_weighted_loss(
           outputs, new_training=self.video_train_cfg['new_training'])

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
            
        return outputs, loss
    
    def remove_params_loss(self, outputs, keeped_loss_names=['CenterMap'], remove_loss_names=['MPJPE', 'PAMPJPE', 'P_KP2D', 'Pose', 'Shape', 'Cam']):
        outputs['loss_dict'] = {
            loss_name: outputs['loss_dict'][loss_name] for loss_name in keeped_loss_names if loss_name in outputs['loss_dict']}
        return outputs

    def train_log_visualization(self, outputs, loss, run_time, data_time, losses, losses_dict, epoch, iter_index):
        losses.update(loss.item())
        losses_dict.update(outputs['loss_dict'])
        if self.global_count%self.print_freq==0:
            message = 'Epoch: [{0}][{1}/{2}] Time {data_time.avg:.2f} RUN {run_time.avg:.2f} Lr {lr} Loss {loss.avg:.2f} | Losses {3}'.format(
                      epoch, iter_index + 1,  len(self.loader), losses_dict.avg(), #Acc {3} | accuracies.avg(), 
                      data_time=data_time, run_time=run_time, loss=losses, lr = self.optimizer.param_groups[0]['lr'])
            print(message)
            write2log(self.log_file,'%s\n' % message)
            self.summary_writer.add_scalar('loss', losses.avg, self.global_count)
            self.summary_writer.add_scalars('loss_items', losses_dict.avg(), self.global_count)
            
            losses.reset(); losses_dict.reset(); data_time.reset() #accuracies.reset(); 
            self.summary_writer.flush()

        if self.global_count%(4*self.print_freq)==0 or self.global_count==1:
            #vis_ids, vis_errors = determ_worst_best(outputs['kp_error'], top_n=3)
            save_name = '{}'.format(self.global_count)
            for ds_name in set(outputs['meta_data']['data_set']):
                save_name += '_{}'.format(ds_name)
            # 'motion_offset','kp2d_gt','pj2d','centermap3D', 'org_img', , 'centermap'    'vids': vis_ids, 'verrors': [vis_errors], 
            # ,'centermap' take care of drop first frame while learning image
            self.visualizer.visulize_video_result(outputs, outputs['meta_data'], show_items=['mesh','motion_offset','centermap'],\
                vis_cfg={'settings': ['save_img'], 'save_dir':self.train_img_dir, 'save_name':save_name, 'error_names':['E']})

    def train_epoch(self, epoch):
        run_time, data_time, losses = [AverageMeter() for i in range(3)]
        losses_dict= AverageMeter_Dict()
        batch_start_time = time.time()

        for iter_index, meta_data in enumerate(self.loader):
            self.global_count += 1
            if args().new_training and self.global_count==args().new_training_iters:
                self.video_train_cfg['new_training'], self.video_eval_cfg['new_training'] = False, False

            data_time.update(time.time() - batch_start_time)
            run_start_time = time.time()
            meta_data = flatten_clip_data(meta_data)

            if check_input_data_quality(meta_data):
                outputs, loss = self.train_step(meta_data)
                if self.local_rank in [-1, 0]:
                    run_time.update(time.time() - run_start_time)
                    self.train_log_visualization(outputs, loss, run_time, data_time, losses, losses_dict, epoch, iter_index)
            
            if self.global_count%self.test_interval==0 or self.global_count==self.fast_eval_iter: #self.print_freq*2
                title='{}_val_{}'.format(self.tab, self.global_count)
                self.save_all_models(title)
            
            if self.distributed_training:
                # wait for rank 0 process finish the job
                torch.distributed.barrier()
            batch_start_time = time.time()
            
        title  = '{}_epoch_{}'.format(self.tab,epoch)
        self.save_all_models(title)
        self.e_sche.step()
        # re-sampling the video clips from the complete video sequence.
        self.loader.dataset.resampling_video_clips()
        
    
    def save_all_models(self, title, ext='.pkl', parent_folder=None):
        parent_folder = self.model_save_dir if parent_folder is None else parent_folder
        save_model(self.temporal_model,title+ext,parent_folder=parent_folder)
        if args().train_backbone:
            save_model(self.image_model,title+'_backbone'+ext,parent_folder=parent_folder)
    
    def train(self):
        # Speed-reproducibility tradeoff: 
        # cuda_deterministic=False is faster but less reproducible, cuda_deterministic=True is slower but more reproducible
        init_seeds(self.local_rank, cuda_deterministic=False)
        logging.info('start training')
        self.temporal_model.train()
        for epoch in range(self.epoch):
            self.train_epoch(epoch)
        self.summary_writer.close()

def check_input_data_quality(meta_data, min_num=2):
    #每个堆里必须要有一个可以监督CenterMap的数据，不然检测的部分loss会崩 nan.
    return meta_data['all_person_detected_mask'].sum()>min_num


def flatten_clip_data(meta_data):
    seq_num, clip_length = meta_data['image'].shape[:2]
    key_names = list(meta_data.keys())
    
    for key in key_names:
        if isinstance(meta_data[key], torch.Tensor):
            shape_list = meta_data[key].shape
            if len(shape_list)>2:
                #print(key,'before',meta_data[key].shape)
                meta_data[key] = meta_data[key].view(-1,*shape_list[2:])
                #print(key,'after',meta_data[key].shape)
            else: 
                meta_data[key] = meta_data[key].view(-1)
            #print(key, shape_list, meta_data[key].shape)
        elif isinstance(meta_data[key], list):
            # the list would be in shape (clip_length, batch_size), so we have to transpose the dimension to (batch_size, clip_length) to make it right.
            meta_data[key] = list(np.array(meta_data[key]).transpose(1,0).reshape(-1))
            #print(meta_data[key])
    # the index of data in format [sequence ID, clip frame ID, flatten batch ID, whether sequence flag]
    meta_data['seq_inds'] = torch.stack([torch.arange(seq_num).unsqueeze(1).repeat(1,clip_length).reshape(-1),\
                                        torch.arange(clip_length).unsqueeze(0).repeat(seq_num,1).reshape(-1),\
                                        torch.arange(seq_num*clip_length),\
                                        torch.ones(seq_num*clip_length)], 1).long()
    return meta_data

def concat_sequence_image_data(seq_data, img_data, del_keys=['seq_inds'], repeat_time=args().image_repeat_time):
    keys = list(seq_data.keys())
    for key in del_keys:
        keys.remove(key)
    image_num = len(img_data['image'])

    mixing_inds = torch.arange(repeat_time*image_num).reshape(repeat_time,-1).transpose(1,0).reshape(-1)
    # print('seq keys', keys, '\n img keys', list(img_data.keys()))
    for key in keys:
        if isinstance(seq_data[key], torch.Tensor):
            # we need to double the image to form the temporal clip with 2 frames
            img_info = torch.cat([img_data[key] for _ in range(repeat_time)], 0)[mixing_inds]
            seq_data[key] = torch.cat([seq_data[key], img_info], 0)
            #print(key, seq_data[key].shape)
        elif isinstance(seq_data[key], list):
            img_info = img_data[key]
            for _ in range(repeat_time-1):
                img_info = img_info + img_data[key]
            img_info = [img_info[i] for i in mixing_inds]
            seq_data[key] = seq_data[key] + img_info
            #print(key, seq_data[key])
    
    seq_num = image_num
    clip_length = repeat_time
    img_seq_inds = torch.stack([max(seq_data['seq_inds'][:,0])+1+torch.arange(seq_num).unsqueeze(1).repeat(1,clip_length).reshape(-1),\
                                        torch.arange(clip_length).unsqueeze(0).repeat(seq_num,1).reshape(-1),\
                                        max(seq_data['seq_inds'][:,2])+1+torch.arange(seq_num*clip_length),\
                                        torch.zeros(seq_num*clip_length)], 1)

    seq_data['seq_inds'] = torch.cat([seq_data['seq_inds'], img_seq_inds],0).long()
    return seq_data

def input2local_rank(meta_data, local_rank):
    keys = list(meta_data.keys())
    for key in keys:
        if isinstance(meta_data[key], torch.Tensor):
            meta_data[key] = meta_data[key].to(f'cuda:{local_rank}')
    return meta_data

class data_iter(object):
    def __init__(self, loader):
        self.loader = loader
        self.loader_iter = iter(loader)
    def next(self):
        try:
            return next(self.loader_iter)
        except:
            self.loader_iter = iter(self.loader)
            return next(self.loader_iter)

def main():
    with ConfigContext(parse_args(sys.argv[1:])):
        trainer = Trainer()
        trainer.train()
        
if __name__ == '__main__':
    main()