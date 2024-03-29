import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F

from lib.tracker.kalman_filter_3dcenter import KalmanFilter
from lib.tracker import matching
from lib.tracker.basetrack import BaseTrack, TrackState
from lib.config import args

class Tracker(object):
    def __init__(self, det_thresh=0.05, first_frame_det_thresh=0.12, match_thresh=1., \
                        accept_new_dets=False, new_subject_det_thresh=0.8,\
                        axis_times=np.array([1.1, 0.9, 10]), track_buffer=60, frame_rate=30):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.match_thresh = match_thresh
        self.det_thresh = det_thresh
        self.first_frame_det_thresh = first_frame_det_thresh
        self.new_subject_det_thresh = 0.8
        self.accept_new_dets = accept_new_dets
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        
        self.max_time_lost = self.buffer_size
        if args().tracking_with_kalman_filter:
            self.kalman_filter = KalmanFilter()
        self.duplicat_dist_thresh = 0.66
        self.axis_times = axis_times

    def update(self, trans3D, scores, last_trans3D, czyxs, \
                debug=True, never_forget=False, tracking_target_max_num=100, \
                using_motion_offsets=True):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if self.frame_id == 1:
            remain_inds = scores > self.first_frame_det_thresh
        else:
            remain_inds = scores > self.det_thresh
        dets = trans3D[remain_inds]
        scores_keep = scores[remain_inds]
        last_dets_keep = last_trans3D[remain_inds]
        czyxs_keep = czyxs[remain_inds]

        if len(dets) > 0:
            # to avoid the small scale shaking cause large depth difference, we use  the inverse of depth for tracking
            # 1/(z+1) change less rapid between (0~1), which make it much safer and stable to use.
            detections = [STrack3D(np.array([*trans[:2], 1/(1+trans[2])]), s, czyx) for
                        (trans, s, czyx) in zip(dets, scores_keep, czyxs_keep)]
            # offset is 3D offset vector from 3D translation of previous frame to current frame
            detections_add_offsets = [STrack3D(np.array([*trans[:2], 1/(1+trans[2])]), s, czyx) for
                        (trans, s, czyx) in zip(last_dets_keep, scores_keep, czyxs_keep)]
        else:
            detections = []
            detections_add_offsets = []

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = self.tracked_stracks
        if debug:
            print(f'___________________{self.frame_id}________________________')
            if len(strack_pool)>0:
                print('strack_pool')
                print(np.stack([np.array([*track.trans, track.track_id, track.score]) for track in strack_pool]))
            if len(detections_add_offsets)>0:
                print('detections_add_offsets')
                print(np.stack([np.array([*track.trans, track.track_id, track.score]) for track in detections_add_offsets]))
        if using_motion_offsets:
            dists = euclidean_distance(strack_pool, detections_add_offsets, dim=3, aug=self.axis_times)
        else:
            dists = euclidean_distance(strack_pool, detections, dim=3, aug=self.axis_times)
            #print('1 dist w/ w/o motion offsets',np.stack([dists, dists_withoutoffsets], 1))

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_add_offsets[idet]
            if track.state == TrackState.Tracked:
                track.update(detections_add_offsets[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            #print(itracked, idet, track.trans, track.track_id)
        
        for it in u_track:
            track = self.tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
            lost_stracks.append(track)

        """ Step 4: Init new stracks with the strack is empty, like the first frame"""
        if not self.accept_new_dets and len(self.tracked_stracks)<tracking_target_max_num:
            u_detection = np.array(u_detection)
            track_scores = np.array([detections[inew].score for inew in u_detection])
            scale_ok_mask = track_scores > self.first_frame_det_thresh
            u_detection = u_detection[scale_ok_mask]
            track_scales = np.array([detections[inew]._trans[2] for inew in u_detection])
            max_scale_subject_inds = u_detection[np.argsort(track_scales)[::-1][:tracking_target_max_num]]
            #print('max_scale_subject_inds', max_scale_subject_inds, track_scales, track_scores, u_detection)
            for inew in max_scale_subject_inds:
                track = detections[inew]
                track.activate(self.frame_id)
                activated_starcks.append(track)
        elif self.accept_new_dets:
            for inew in u_detection:
                track = detections[inew]
                if len(strack_pool) ==0 and track.score > self.first_frame_det_thresh:
                    track.activate(self.frame_id)
                    activated_starcks.append(track)
                elif track.score > self.new_subject_det_thresh:
                    track.activate(self.frame_id)
                    activated_starcks.append(track)
        
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        
        """ Step 5: Update state"""
        for track in lost_stracks:
            if debug:
                print('lost track:', track.trans, self.frame_id, track.end_frame, self.max_time_lost)
            if self.frame_id - track.end_frame > self.max_time_lost and not never_forget:
                track.mark_removed()
                removed_stracks.append(track)
                self.tracked_stracks = sub_stracks(self.tracked_stracks, [track])

        output_results = np.array([np.array([*track.trans[:3], track.track_id, track.score, track.state == TrackState.Tracked, *track.czyx]) \
                            for track in self.tracked_stracks if track.is_activated])
        if debug:
            print(output_results)

        return copy.deepcopy(output_results)
    
    def update_BT(self, trans3D, scores, last_trans3D, czyxs, debug=True, det_thresh=0.12, low_conf_det_thresh=0.05, match_thresh=1):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        remain_inds = scores > det_thresh
        dets = trans3D[remain_inds]
        scores_keep = scores[remain_inds]
        last_dets_keep = trans3D[remain_inds]
        czyxs_keep = czyxs[remain_inds]

        inds_second = np.logical_and(scores > low_conf_det_thresh, scores < det_thresh)
        dets_second = trans3D[inds_second]
        scores_second = scores[inds_second]
        last_dets_second = trans3D[inds_second]
        czyxs_second = czyxs[inds_second]

        if len(dets) > 0:
            detections = [STrack(np.array([*trans, 1/(1+trans[2])]), s, czyx) for
                          (trans, s, czyx) in zip(dets, scores_keep, czyxs_keep)]
        else:
            detections = []
            detections_add_offsets = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        dists = matching.euclidean_distance(strack_pool, detections, dim=4)
            
        #dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(np.concatenate([trans, np.array([1/(1+trans[2])])], 0), s, czyx) for
                          (trans, s, czyx) in zip(dets_second, scores_second, czyxs_second)]
        else:
            detections_second = []
            detections_add_offsets_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.euclidean_distance(r_tracked_stracks, detections_second, dim=4)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=match_thresh*2)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.euclidean_distance(unconfirmed, detections)
        # dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=match_thresh*3)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if debug:
                print('lost track:', track.trans, self.frame_id, track.end_frame, self.max_time_lost)
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)

        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks, dist_thresh=self.duplicat_dist_thresh)

        output_results = np.array([np.array([*track.trans[:3], track.track_id, track.score, track.state == TrackState.Tracked, *track.czyx]) \
                            for track in self.tracked_stracks if track.is_activated])
        if debug:
            print(output_results)

        return copy.deepcopy(output_results)


def euc_dist(atrans, btrans):
    euc_dists = np.zeros((len(atrans), len(btrans)), dtype=np.float32)
    if euc_dists.size == 0:
        return euc_dists
    euc_dists = np.linalg.norm(np.array(atrans)[:, None]-np.array(btrans)[None], ord=2, axis=2)
    return euc_dists


def euclidean_distance(atracks, btracks, dim=4, aug=np.array([1,1,1])):
    """
    TO handle the fast scale change of the close object, 
    Compute euclidean distance between 3D body centers + bbox height
    :type atracks: list[STrack]
    :type btracks: list[STrack]
    :rtype cost_matrix np.ndarray
    """
    atrans = np.array([copy.deepcopy(track.trans[:dim]) for track in atracks])
    btrans = np.array([copy.deepcopy(track.trans[:dim]) for track in btracks])
    if len(atrans)>0:
        atrans = atrans*aug[None]
    if len(btrans)>0:
        btrans = btrans*aug[None]
    cost_matrix = euc_dist(atrans, btrans)

    return cost_matrix

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb, dist_thresh=0.15):
    # only use the 3D points for suppression. 2D points would make the occluded person disapear
    pdist = matching.euclidean_distance(stracksa, stracksb, dim=2)
    pairs = np.where(pdist < dist_thresh)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

def remove_duplicate_stracks3D(stracksa, stracksb, dist_thresh=0.15):
    # only use the 3D points for suppression. 2D points would make the occluded person disapear
    pdist = matching.euclidean_distance(stracksa, stracksb, dim=3)
    pairs = np.where(pdist < dist_thresh)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    if len(dupa)>0 or len(dupb)>0:
        print('duplicate_stracks:', dupa, dupb, pdist)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


class STrack3D(BaseTrack):
    def __init__(self, trans, score, czyx):
        # wait activate
        self._trans = np.asarray(trans, dtype=np.float32)
        self.is_activated = False
        self.score = score
        self.czyx = czyx
        self.tracklet_len = 0

    def activate(self, frame_id):
        """Start a new tracklet"""
        self.track_id = self.next_id()
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        self._trans = new_track.trans
        self.czyx = new_track.czyx
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def trans(self):
        """Get current 3D body center position `(x,y,z)`.
        """
        return self._trans.copy()

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)
    

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, trans, score, czyx):
        # wait activate
        self._trans = np.asarray(trans, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.czyx = czyx
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            ''' don't know why doing this '''
            # 可以考虑被遮挡的目标的状态跟随遮挡者的速度状态。形成遮挡跟随模式，直到被遮挡者再次出现。
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0

            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self._trans)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.trans
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_trans = new_track.trans
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_trans)
        self.state = TrackState.Tracked
        self.is_activated = True

        self.czyx = new_track.czyx
        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def trans(self):
        """Get current 3D body center position `(x,y,z)`.
        """
        if self.mean is None:
            return self._trans.copy()
        ret = self.mean[:4].copy()
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)