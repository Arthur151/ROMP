import numpy as np
from numba import jit
from collections import deque
import itertools
import os
import os.path as osp
import time
import torch
import cv2
import torch.nn.functional as F
import sys, os

from tracking.tracking_utils.utils import *
from tracking.tracking_utils.log import logger
from tracking.tracking_utils.kalman_filter import KalmanFilter
from tracking import matching
from tracking.basetrack import BaseTrack, TrackState
#from utils.post_process import ctdet_post_process


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, trans_uv, body_pose, conf, buffer_size=30):

        # wait activate
        self._centers = np.asarray(trans_uv, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = conf
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_poses(body_pose)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

    def update_poses(self, body_pose):
        self.body_pose = body_pose
        '''
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)
        '''

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
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][5] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.center)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.center
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_center = new_track.center
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_center)
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def center(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return np.concatenate([self._centers.copy(), self.score])
        ret = self.mean[:4].copy()
        return ret


    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class Tracker(object):
    def __init__(self, seq_name='test', frame_rate=30):
        self.seq_name = seq_name
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.buffer_size = int(frame_rate / 30.0 * 30)
        self.max_time_lost = 15 #self.buffer_size
        self.kalman_filter = KalmanFilter()

        self.deal_with_unconfirm = False

    def post_process(self, outputs):
        cams = [result['cam'] for result in outputs]
        kp3ds = [result['j3d_all54'] for result in outputs]
        center_conf = [result['center_conf'] for result in outputs]
        return cams, kp3ds, center_conf

    def update(self, outputs):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        cams, kp3ds, center_conf = self.post_process(outputs)

        if len(cams)>0:
            '''Detections'''
            detections = [STrack(trans_suv, body_pose, conf, 30) for (trans_suv, body_pose, conf) in zip(cams, kp3ds, center_conf)]
            tracked_ids = np.zeros(len(detections))
        else:
            detections = []
            tracked_ids = np.array([0])

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        dists = matching.gate_cost_matrix(self.kalman_filter, strack_pool, detections, only_position=True)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.4)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            tracked_ids[idet] = track.track_id
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        for track in r_tracked_stracks:
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        if self.deal_with_unconfirm:
            '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            detections = [detections[i] for i in u_detection]
            dists = matching.gate_cost_matrix(self.kalman_filter, unconfirmed, detections)
            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
            for itracked, idet in matches:
                unconfirmed[itracked].update(detections[idet], self.frame_id)
                activated_starcks.append(unconfirmed[itracked])
            for it in u_unconfirmed:
                track = unconfirmed[it]
                track.mark_removed()
                removed_stracks.append(track)
        else:
            for track in unconfirmed:
                track.mark_removed()
                removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
            tracked_ids[inew] = track.track_id
        """ Step 5: Update state"""
        for track in self.lost_stracks:
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
        #self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.kalman_filter, self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        # logger.debug('===========Frame {}=========='.format(self.frame_id))
        # logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        # logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        # logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        # logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        online_centers = []
        online_ids = []
        for t in output_stracks:
            online_centers.append(t.center)
            online_ids.append(t.track_id)

        return online_ids


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


def remove_duplicate_stracks(kf,stracksa, stracksb):
    pdist = matching.gate_cost_matrix(kf, stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
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