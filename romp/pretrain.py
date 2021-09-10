from base import *
from eval import val_result
from loss_funcs import Learnable_Loss
from maps_utils import HeatmapParser,CenterMap
from loss_funcs.maps_loss import focal_loss, Heatmap_AE_loss
from loss_funcs.keypoints_loss import batch_kp_2d_l2_loss
from visualization.visualization import draw_skeleton_multiperson, draw_skeleton, make_heatmaps
np.set_printoptions(precision=2, suppress=True)

class Trainer(Base):
    def __init__(self):
        super(Trainer, self).__init__()
        self._build_model_()
        self._build_optimizer()
        self.set_up_validation()
        self.mutli_task_uncertainty_weighted_loss = Learnable_Loss().cuda()
        self.optimizer.add_param_group({'params': self.mutli_task_uncertainty_weighted_loss.parameters()})
        self.loader = self._create_data_loader(train_flag=True)
        if args().learn_2dpose or args().learn_AE:
            self.heatmap_parser = HeatmapParser()
            self.heatmap_aeloss = Heatmap_AE_loss(17, loss_type_HM=args().HMloss_type, loss_type_AE='exp')
            self.centermap_parser = CenterMap()
        self.train_cfg = {'mode':'train', 'update_data': True, 'calc_loss': True if self.model_return_loss else False, \
                           'new_training': Flase}
        self.eval_cfg = {'mode':'train', 'calc_loss': False}
        self.val_cfg = {'mode':'val', 'calc_loss': False}
        
        self.result_save_dir = '/export/home/suny/pretrain_result_images'
        os.makedirs(self.result_save_dir,exist_ok=True)
        logging.info('Initialization of Trainer finished!')

    def set_up_validation(self):
        self.evaluation_results_dict = {}
        self.dataset_val_list, self.dataset_test_list = {}, {}
        self.dataset_val_list['coco'] = self._create_single_data_loader(dataset='coco', train_flag=False, regress_smpl=False)
        self.dataset_val_list['crowdhuman'] = self._create_single_data_loader(dataset='crowdhuman', train_flag=False)
        
        logging.info('dataset_val_list:{}'.format(list(self.dataset_val_list.keys())))

    def _calc_loss(self, outputs, meta_data):
        device = outputs['center_map'].device
        loss_dict = {}
        if 'kp_ae_maps' in outputs:
            loss_dict['heatmap'],loss_dict['AE'] = self.heatmap_AE_loss(meta_data['full_kp2d'].to(device), outputs['kp_ae_maps'], meta_data['heatmap'].to(device), meta_data['AE_joints'])
        all_person_mask = meta_data['all_person_detected_mask'].to(device)
        if all_person_mask.sum()>0:
            loss_dict['CenterMap'] = focal_loss(outputs['center_map'][all_person_mask], meta_data['centermap'][all_person_mask].to(device))

        loss_names = list(loss_dict.keys())
        for name in loss_names:
            if isinstance(loss_dict[name],tuple):
                loss_dict[name] = loss_dict[name][0]
            loss_dict[name] = loss_dict[name].mean() * eval('args.{}_weight'.format(name))

        return {'loss_dict':loss_dict}

    def heatmap_AE_loss(self, real, pred, heatmap_gt, joints):
        #heatmap_gt = self.heatmap_generator.batch_process(real)
        #joints = self.joint_generator.batch_process(real)
        heatmaps_loss, push_loss, pull_loss = self.heatmap_aeloss(pred,heatmap_gt,joints)
        AE_loss = pull_loss + push_loss
        return heatmaps_loss, AE_loss

    def train(self):
        # Speed-reproducibility tradeoff: 
        # cuda_deterministic=False is faster but less reproducible, cuda_deterministic=True is slower but more reproducible
        init_seeds(self.local_rank, cuda_deterministic=False)
        logging.info('start training')
        self.model.train()
        if not self.fine_tune and self.fix_backbone_training_scratch:
            fix_backbone(self.model, exclude_key=['backbone.'])
        for epoch in range(self.epoch):
            if epoch==1:
                train_entire_model(self.model)
            self.train_epoch(epoch)
        self.summary_writer.close()

    def train_epoch(self,epoch):
        run_time, data_time, losses = [AverageMeter() for i in range(3)]
        losses_dict= AverageMeter_Dict()
        batch_start_time = time.time()
        for iter_index, meta_data in enumerate(self.loader):
            self.global_count += 1
            data_time.update(time.time() - batch_start_time)
            run_start_time = time.time()

            self.optimizer.zero_grad()
            if self.model_precision=='fp16':
                with autocast():
                    outputs = self.model(meta_data)
                outputs.update(self._calc_loss(outputs, meta_data))
                loss, outputs = self.mutli_task_uncertainty_weighted_loss(outputs)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(meta_data)
                outputs.update(self._calc_loss(outputs, meta_data))
                loss, outputs = self.mutli_task_uncertainty_weighted_loss(outputs)
                loss.backward()
                self.optimizer.step()

            if self.local_rank in [-1, 0]:
                run_time.update(time.time() - run_start_time)
                losses.update(loss.item())
                losses_dict.update(outputs['loss_dict'])
                if self.global_count%self.print_freq==0:
                    message = 'Epoch: [{0}][{1}/{2}] Time {data_time.avg:.2f} RUN {run_time.avg:.2f} Lr {lr} Loss {loss.avg:.2f} | Losses {3}'.format(
                              epoch,  iter_index + 1,  len(self.loader), losses_dict.avg(), #Acc {3} | accuracies.avg(), 
                              data_time=data_time, run_time=run_time, loss=losses, lr = self.optimizer.param_groups[0]['lr'])
                    print(message)
                    write2log(self.log_file,'%s\n' % message)
                    self.summary_writer.add_scalar('loss', losses.avg, self.global_count)
                    self.summary_writer.add_scalars('loss_items', losses_dict.avg(), self.global_count)
                    
                    losses.reset(); losses_dict.reset(); data_time.reset()
                    self.summary_writer.flush()

            if self.global_count%self.test_interval==0 or (self.global_count==10 and self.fast_eval):
                self.validation(epoch)
            batch_start_time = time.time()
            
        title  = '{}_epoch_{}.pkl'.format(self.tab,epoch)
        save_model(self.model,title,parent_folder=self.model_save_dir)

    def validation(self,epoch):
        if self.distributed_training:
            eval_model = self.model.module
        elif self.master_batch_size!=-1:
            eval_model = nn.DataParallel(self.model.module)
        else:
            eval_model = self.model
        eval_model.eval()
        logging.info('evaluation result on {} iters: '.format(epoch))
        for ds_name, val_loader in self.dataset_val_list.items():
            logging.info('Evaluation on {}'.format(ds_name))
            for iter_index, meta_data in enumerate(val_loader):
                if self.model_precision=='fp16':
                    with autocast():
                        outputs = eval_model(meta_data)
                else:
                    outputs = eval_model(meta_data)
                if args().learn_2dpose or args().learn_AE:
                    self.visualize_outputs_hmae(outputs, meta_data, epoch)

                if iter_index == 5:
                    break

        title = '{}_{}.pkl'.format(epoch, self.tab)
        logging.info('Model saved as {}'.format(title))
        save_model(eval_model,title,parent_folder=self.model_save_dir)

        self.model.train()

    def visualize_outputs_offsets(self, outputs, meta_data, epoch):
        centers_pred = outputs['centers_pred'].cpu().numpy()
        kp2d_preds = outputs['joint_sampler'].detach().cpu().numpy()
        batch_ids = outputs['reorganize_idx'].cpu().numpy()
        for bid in np.unique(batch_ids):
            cids = np.where(batch_ids==bid)[0]
            img_path = outputs['meta_data']['imgpath'][cids[0]]
            if len(cids)==0:
                print('detection failed on {}'.format(img_path))
            img = meta_data['image_org'][bid].numpy()[:,:,::-1]
            centermap_color = make_heatmaps(img.copy(), outputs['center_map'][bid])

            centers = (centers_pred[cids]/float(args().centermap_size) * img.shape[1])
            center_img = draw_skeleton(img.copy(), centers, r=6)
            
            kp2ds = ((kp2d_preds[cids]+1)/2 * img.shape[1]).reshape((len(cids),-1,2))
            skeleton_img = draw_skeleton_multiperson(img.copy(), kp2ds, bones=constants.joint_sampler_connMat, cm=constants.cm_body25)
            
            result_img = np.concatenate([centermap_color, center_img, skeleton_img],1)
            save_name = os.path.join(self.result_save_dir, str(epoch)+os.path.basename(img_path))
            cv2.imwrite(save_name, result_img)

    def visualize_outputs_hmae(self, outputs, meta_data, epoch):
        center_preds_info = self.centermap_parser.parse_centermap(outputs['center_map'])
        batch_ids, topk_inds, center_yxs, topk_score = center_preds_info
        kp2d_preds = self.parse_kps(outputs['kp_ae_maps'])
        for bid, img_path in enumerate(meta_data['imgpath']):
            cids = torch.where(batch_ids==bid)[0]
            centers = center_yxs[cids].cpu().numpy()
            kp2ds = kp2d_preds[bid]
            img = meta_data['image_org'][bid].numpy()[:,:,::-1]
            kp2ds = ((kp2ds+1)/2 * img.shape[1])
            centers = (centers[:,::-1]/float(args().centermap_size) * img.shape[1])
            centermap_color = make_heatmaps(img.copy(), outputs['center_map'][bid])

            skeleton_img = draw_skeleton_multiperson(img.copy(), kp2ds, bones=constants.body17_connMat, cm=constants.cm_body17)
            center_img = draw_skeleton(img.copy(), centers, r=6)

            result_img = np.concatenate([centermap_color, center_img, skeleton_img],1)
            save_name = os.path.join(self.result_save_dir, str(epoch)+os.path.basename(img_path))
            cv2.imwrite(save_name, result_img)

    def parse_kps(self, heatmap_AEs, kp2d_thresh=0.1):
        kp2ds, scores_each_person = self.heatmap_parser.batch_parse(heatmap_AEs.detach())
        return kp2ds

def main():
    with ConfigContext(parse_args(sys.argv[1:])):
        trainer = Trainer()
        trainer.train()

if __name__ == '__main__':
    main()

