"""
Modified from https://github.com/brjathu/PHALP/blob/c60e6807a97d6e2b74f48bfbad8c95a693435a87/evaluate_PHALP.py
"""

import numpy as np
import os, sys, glob, cv2
try:
    import motmetrics as mm
except:
    os.system("pip install motmetrics")
    import motmetrics as mm
import joblib
from .TrackEval import trackeval

def prepare_data(results,data_gt):
    data_all              = {}
    total_annoated_frames = 0
    total_detected_frames = 0
    
    for video_ in data_gt.keys():
        predictions = results[video_]
        list_of_gt_frames = np.sort(list(data_gt[video_].keys()))
        tracked_frames    = list(predictions.keys())
        data_all[video_]  = {}
        for i in range(len(list_of_gt_frames)):
            frame_        = list_of_gt_frames[i]
            total_annoated_frames += 1
            if(frame_ in tracked_frames):
                tracked_data = predictions[frame_]
                if(len(data_gt[video_][frame_][0])>0):
                    assert data_gt[video_][frame_][0][0].split("/")[-1] == frame_
                    if(len(tracked_data['track_ids'])==0):   
                        data_all[video_][frame_] = [data_gt[video_][frame_][0], data_gt[video_][frame_][1], data_gt[video_][frame_][2], data_gt[video_][frame_][3], [], [], []]
                    else:
                        data_all[video_][frame_] = [data_gt[video_][frame_][0], data_gt[video_][frame_][1], data_gt[video_][frame_][2], data_gt[video_][frame_][3], frame_, \
                            tracked_data['track_ids'], tracked_data['track_bbox']] 
                        total_detected_frames += 1
                    
            else:
                #print(frame_, 'missing')
                data_all[video_][frame_] = [data_gt[video_][frame_][0], data_gt[video_][frame_][1], data_gt[video_][frame_][2], data_gt[video_][frame_][3], [], [], []]; 

    return data_all

def _from_dense(num_timesteps, num_gt_ids, num_tracker_ids, gt_present, tracker_present, similarity):
    gt_subset = [np.flatnonzero(gt_present[t, :]) for t in range(num_timesteps)]
    tracker_subset = [np.flatnonzero(tracker_present[t, :]) for t in range(num_timesteps)]
    similarity_subset = [
            similarity[t][gt_subset[t], :][:, tracker_subset[t]]
            for t in range(num_timesteps)
    ]
    data = {
            'num_timesteps': num_timesteps,
            'num_gt_ids': num_gt_ids,
            'num_tracker_ids': num_tracker_ids,
            'num_gt_dets': np.sum(gt_present),
            'num_tracker_dets': np.sum(tracker_present),
            'gt_ids': gt_subset,
            'tracker_ids': tracker_subset,
            'similarity_scores': similarity_subset,
    }
    return data

def accumulate_results(data_all):
    accumulators = []   
    TOTAL_AssA   = []
    TOTAL_DetA   = []
    TOTAL_HOTA   = []
    for video in list(data_all.keys()):
        acc = mm.MOTAccumulator(auto_id=True)
        accumulators.append(acc)

        # evaluate HOTA
        T                = len(data_all[video].keys())
        gt_ids_hota      = np.zeros((T, 500))
        pr_ids_hota      = np.zeros((T, 500))
        similarity_hota  = np.zeros((T, 500, 500))
        gt_available     = []   
        hota_metric      = trackeval.metrics.HOTA()
        start_           = 0
        
        list_of_predictions  = []
        for t, frame in enumerate(data_all[video].keys()):
            data        = data_all[video][frame]
            pt_ids      = data[5]
            for p_ in pt_ids:
                list_of_predictions.append(p_)
        list_of_predictions = np.unique(list_of_predictions)   
        
        for t, frame in enumerate(data_all[video].keys()):
            data = data_all[video][frame]
            gt_ids      = data[1]
            gt_ids_new  = data[3]
            gt_bbox     = data[2]
            pt_ids_      = data[5]
            pt_bbox_     = data[6]

            pt_ids  = []
            pt_bbox = []
            for p_, b_ in zip(pt_ids_, pt_bbox_):
                loc= np.where(list_of_predictions==p_)[0][0]
                pt_ids.append(loc)
                pt_bbox.append(b_)

            if(len(gt_ids_new)>0):
                cost_ = mm.distances.iou_matrix(gt_bbox, pt_bbox, max_iou=0.99)
                #cost_[np.isnan(cost_)] = 100.
                accumulators[-1].update(
                                                    gt_ids_new,  # Ground truth objects in this frame
                                                    pt_ids,      # Detector hypotheses in this frame
                                                    cost_)
                cost_[np.isnan(cost_)] = 1
                ############# HOTA evaluation code
                gt_available.append(t)
                for idx_gt in gt_ids_new:
                    gt_ids_hota[t][idx_gt] = 1

                for idx_pr in pt_ids:
                    pr_ids_hota[t][idx_pr] = 1

                for i, idx_gt in enumerate(gt_ids_new):
                    for j, idx_pr in enumerate(pt_ids):
                        similarity_hota[t][idx_gt][idx_pr] = 1-cost_[i][j]
            
        gt_ids_hota     = gt_ids_hota[gt_available, :]
        pr_ids_hota     = pr_ids_hota[gt_available, :]
        similarity_hota = similarity_hota[gt_available, :]

        data = _from_dense(
                num_timesteps  =  len(gt_available),
                num_gt_ids     =  np.sum(np.sum(gt_ids_hota, 0)>0),
                num_tracker_ids=  np.sum(np.sum(pr_ids_hota, 0)>0),
                gt_present     =  gt_ids_hota,
                tracker_present=  pr_ids_hota,
                similarity     =  similarity_hota,
        )
        
        results = hota_metric.eval_sequence(data)    
        TOTAL_AssA.append(np.mean(results['AssA']))
        TOTAL_DetA.append(np.mean(results['DetA']))
        TOTAL_HOTA.append(np.mean(results['HOTA']))
        
    return accumulators, TOTAL_AssA, TOTAL_DetA, TOTAL_HOTA

def evaluate_trackers(results, tracking_matrix_save_path, dataset="posetrack", method="TRACE", gt_path=None):   
    if(dataset=="mupots"):    data_gt = joblib.load('/home/yusun/data/t3dp_tracking_gts/mupots_gt.pickle')
    if(dataset=="Dyna3DPW"):  data_gt = np.load('/home/yusun/DataCenter2/datasets/Dyna3DPW/tracking_gts.npz',allow_pickle=True)['annots'][()]; mot_gt_folder='data/t3dp_tracking_gts/Dyna3DPW'
    if(dataset=="cmup"):  data_gt = np.load('/home/yusun/DataCenter/datasets/CMU_Panoptic_CRMH/packed_tracking_annots.npz',allow_pickle=True)['annots'][()]

    data_all = prepare_data(results, data_gt)

    accumulators, TOTAL_AssA, TOTAL_DetA, TOTAL_HOTA = accumulate_results(data_all)

    mh = mm.metrics.create()

    summary = mh.compute_many(
        accumulators,
        metrics=mm.metrics.motchallenge_metrics,
        generate_overall=True)

    IDF1        = summary['idf1']['OVERALL'] * 100.
    ID_switches = summary['num_switches']['OVERALL']
    MOTA        = summary['mota']['OVERALL'] * 100.
    PRCN        = summary['precision']['OVERALL']
    RCLL        = summary['recall']['OVERALL']

    strsummary  = mm.io.render_summary(
        summary,
        formatters = mh.formatters,
        namemap    = mm.io.motchallenge_metric_names)

    results_ID_switches       = summary['num_switches']['OVERALL']
    results_mota              = summary['mota']['OVERALL'] * 100.
    
    AssA              = np.mean(TOTAL_AssA) * 100
    DetA              = np.mean(TOTAL_DetA) * 100
    HOTA              = np.mean(TOTAL_HOTA) * 100 
    
    print("ID switches  ", results_ID_switches)
    print("MOTA         ", results_mota)
    print("IDF1         ", IDF1)             
    print("HOTA         ", HOTA)    
        
    text_file = open(tracking_matrix_save_path, "w")
    n = text_file.write(strsummary)
    text_file.close()

    ED_results = {'{}-MOTA'.format(dataset): MOTA,
                  '{}-IDs'.format(dataset): ID_switches,
                  '{}-HOTA'.format(dataset): HOTA}

    return ED_results

def pj2ds_to_bbox(pj2ds, enlarge_xy=np.array([1.1,1.18])): # enlarge_xy=np.array([1.1,1.18]) used for evaluation on MuPoTS
    tracked_bbox = np.array([pj2ds[:,0].min(), pj2ds[:,1].min(), pj2ds[:,0].max(), pj2ds[:,1].max()])
    # left, top, right, down -> left, top, width, height
    center = (tracked_bbox[2:] + tracked_bbox[:2]) / 2
    tracked_bbox[2:] = (center - tracked_bbox[:2]) * enlarge_xy
    tracked_bbox[:2] = center - tracked_bbox[2:]
    tracked_bbox[2:] = tracked_bbox[2:] * 2
    #tracked_bbox[2:] = tracked_bbox[2:] - tracked_bbox[:2]
    return tracked_bbox

def adjust_tracking_results(results, enlarge_xy=np.array([1.1,1.18])):
    for seq_name in list(results.keys()):
        tracked_frames = sorted(list(results[seq_name].keys()))
        for frame_name in tracked_frames:
            for ind, (bbox, pj2ds) in enumerate(zip(results[seq_name][frame_name]['track_bbox'], results[seq_name][frame_name]['pj2ds'])):
                results[seq_name][frame_name]['track_bbox'][ind] = pj2ds_to_bbox(pj2ds,enlarge_xy=enlarge_xy)
    return results

def eval_tracking_metrics(tracking_results, data_gt):
    data_all = prepare_data(tracking_results, data_gt)
    accumulators, TOTAL_AssA, TOTAL_DetA, TOTAL_HOTA = accumulate_results(data_all)

    mh = mm.metrics.create()
    summary = mh.compute_many(
        accumulators,
        metrics=mm.metrics.motchallenge_metrics,
        generate_overall=True)

    IDF1        = summary['idf1']['OVERALL'] * 100.
    ID_switches = summary['num_switches']['OVERALL']
    MOTA        = summary['mota']['OVERALL'] * 100.
    PRCN        = summary['precision']['OVERALL']
    RCLL        = summary['recall']['OVERALL']

    strsummary  = mm.io.render_summary(
        summary,
        formatters = mh.formatters,
        namemap    = mm.io.motchallenge_metric_names)

    print(strsummary)
    results_ID_switches       = summary['num_switches']['OVERALL']
    results_mota              = summary['mota']['OVERALL'] * 100.
    
    AssA              = np.mean(TOTAL_AssA) * 100
    DetA              = np.mean(TOTAL_DetA) * 100
    HOTA              = np.mean(TOTAL_HOTA) * 100 
    print("ID switches  ", results_ID_switches)
    print("MOTA         ", results_mota)
    print("IDF1         ", IDF1)           
    print("HOTA         ", HOTA)    
    return MOTA, ID_switches, HOTA, strsummary

def evaluate_trackers_mupots(results_dir, dataset_dir, **kwargs):   
    data_gt = joblib.load(os.path.join(dataset_dir, 'mupots_gt.pickle'))

    tracking_results = {}
    for results_path in glob.glob(os.path.join(results_dir, 'TS*_tracking.npz')):
        seq_tracking_results = np.load(results_path, allow_pickle=True)['tracking'][()]
        seq_name = os.path.basename(results_path).replace('_tracking.npz', '')
        tracking_results[seq_name] = seq_tracking_results
    tracking_results = adjust_tracking_results(tracking_results,enlarge_xy=np.array([1.1,1.18]))

    MOTA, ID_switches, HOTA, strsummary = eval_tracking_metrics(tracking_results, data_gt)
    
    text_file = open(os.path.join(results_dir, 'MuPoTS_tracking_metrics.txt'), "w")
    n = text_file.write(strsummary)
    text_file.close()
    ED_results = {'MuPoTS-MOTA': MOTA, 'MuPoTS-IDs': ID_switches, 'MuPoTS-HOTA': HOTA}

    return ED_results  

Dyna3DPW_seq_names = ['downtown_cafe_00', 'downtown_walkBridge_01', 'downtown_runForBus_01', 'downtown_sitOnStairs_00', \
                    'downtown_bar_00', 'downtown_enterShop_00', 'downtown_rampAndStairs_00', 'downtown_runForBus_00', \
                    'downtown_car_00',  'downtown_walking_00', 'downtown_crossStreets_00', 'office_phoneCall_00', 
                    'downtown_warmWelcome_00', 'downtown_weeklyMarket_00', 'downtown_bus_00', 'downtown_windowShopping_00']
#not_used_dyna3dpw_seqs = ['downtown_upstairs_00', 'downtown_downstairs_00']
def evaluate_trackers_dyna3dpw(results_dir, dataset_dir, **kwargs):   
    data_gt = np.load(os.path.join(dataset_dir, 'dyna3dpw_tracking_gts.npz'), allow_pickle=True)['annots'][()]

    tracking_results = {}
    for seq_name in Dyna3DPW_seq_names:
        results_path = os.path.join(results_dir, f'{seq_name}_tracking.npz')
        seq_tracking_results = np.load(results_path, allow_pickle=True)['tracking'][()]
        tracking_results[seq_name] = seq_tracking_results
    tracking_results = adjust_tracking_results(tracking_results,enlarge_xy=np.array([1.0,1.05]))
    
    MOTA, ID_switches, HOTA, strsummary = eval_tracking_metrics(tracking_results, data_gt)

    text_file = open(os.path.join(results_dir, 'Dyna3DPW_tracking_metrics.txt'), "w")
    n = text_file.write(strsummary)
    text_file.close()

    ED_results = {'Dyna3DPW-MOTA': MOTA, 'Dyna3DPW-IDs': ID_switches, 'Dyna3DPW-HOTA': HOTA}

    return ED_results  

def vis_track_bbox(image_path, tracked_ids, tracked_bbox):
    org_img = cv2.imread(image_path)
    for tid, bbox in zip(tracked_ids, tracked_bbox):
        org_img = cv2.rectangle(org_img, tuple(bbox[:2]), tuple(bbox[2:]+bbox[:2]), (255,0,0), 3)
        org_img = cv2.putText(org_img, "{}".format(tid), tuple(bbox[:2]), cv2.FONT_HERSHEY_COMPLEX, 3, (0,255,0),2)
    h,w = org_img.shape[:2]
    cv2.imshow('bbox', cv2.resize(org_img, (w//2, h//2)))
    cv2.waitKey(5)

dyna3dpw_start_frame = {
    'downtown_bar_00': 66, 
    'downtown_cafe_00': 128, 
    'downtown_car_00': 0,
    'downtown_downstairs_00': 0,
    'downtown_enterShop_00': 0,
    'downtown_rampAndStairs_00': 0,
    'downtown_runForBus_00': 88,
    'downtown_runForBus_01': 300,
    'downtown_sitOnStairs_00': 94,
    'downtown_walkBridge_01': 156,
    'downtown_walking_00': 0,
    'downtown_warmWelcome_00': 0, 
    'downtown_weeklyMarket_00': 0, 
    'downtown_windowShopping_00': 0, 
    'downtown_crossStreets_00': 152,
    'office_phoneCall_00': 0,
    'downtown_bus_00': 0,}

if __name__ == '__main__':
    #eval_bytetrack_mupots()
    #eval_trace_mupots()
    #eval_bytetrack_dyna3dpw()
    #eval_trace_dyna3dpw()
    #eval_bev_bytetrack_dyna3dpw()
    #load_phalp_results()

    pack_cmup_tracking_gts('/home/yusun/DataCenter/datasets/CMU_Panoptic_CRMH/packed_tracking_annots.npz', visualize_results=False)