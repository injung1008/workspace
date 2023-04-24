import numpy as np
# from numba import jit
from .basetrack import STrack, TrackState
from . import matching

def create_strack(detect_results, tracking_info) :
    """ detection결과를 STrack으로 변환

    Args:
        detect_result (_type_): _description_
        track_thres (float, optional): _description_. Defaults to 0.6.

    Returns:
        _type_: _description_
    """
    scores = detect_results[:, 4]
    bboxes = detect_results[:, :4]
    
    # detect result -> STrack
    detect_results = [{"bbox" : bbox, "score" : score, "order" : index} for index, (bbox, score) in enumerate(zip(bboxes, scores))]
    
    # 기억할 사람들
    remain_ids = np.where(scores >= tracking_info.track_thres)[0]
    if len(remain_ids) > 0 :
        remain_results = zip(*[
            (STrack(STrack.tlbr_to_tlwh(detect_results[id]['bbox']), detect_results[id]['score']), detect_results[id]) for id in remain_ids
        ])
    else :
        remain_results = [[], []]
    
    # 임시 기억할 사람들
    inds_low = scores >= 0.1
    inds_high = scores < tracking_info.track_thres
    second_ids = np.where(np.logical_and(inds_low, inds_high))[0]
    if len(second_ids) > 0 :
        second_results = zip(*[
            (STrack(STrack.tlbr_to_tlwh(detect_results[id]['bbox']), detect_results[id]['score']), detect_results[id]) for id in second_ids
        ])
    else :
        second_results = [[], []]
    
    # 사용하지 않는 사람들
    inds_not = np.where(scores < 0.1)[0]
    not_results = [
        (STrack(STrack.tlbr_to_tlwh(detect_results[id]['bbox'], detect_results[id]['score'])), detect_results[id]) for id in inds_not
    ]
    
    return remain_results, second_results, not_results

def track_info_select(track_info) :
    unconfirmed = []
    tracked_stracks = []
    for track in track_info.tracked_stracks :
        if not track.is_activated:
            unconfirmed.append(track)
        else:
            tracked_stracks.append(track)
    
    strack_pool = joint_stracks(tracked_stracks, track_info.lost_stracks)
    STrack.multi_predict(strack_pool)
    return strack_pool, unconfirmed

def matching_fuse_iou(strack_pool, det_stracks, threshold, frame_id, det_results=None) :
    dists = matching.iou_distance(strack_pool, det_stracks)
    dists = matching.fuse_score(dists, det_stracks)
    matches, u_track, u_detection = matching.linear_assignment(dists, threshold)
    
    matched_tracks, matched_results = [], []
    for i_track, i_det in matches :
        track = strack_pool[i_track]
        det_track = det_stracks[i_det]
        if track.state == TrackState.Tracked :
            track.update(det_track, frame_id)
        else :
            track.re_activate(det_track, frame_id, new_id=False)
        
        if det_results is not None :
            track_result = det_results[i_det]
            track_result['id'] = track.track_id
            track_result['bbox'] = track.tlbr
            matched_results.append(track_result)
        
        matched_tracks.append(track)
    
    u_strack_pool = [strack_pool[i_track] for i_track in u_track]
    u_det_stracks = [det_stracks[i_det] for i_det in u_detection]
    u_matched_results = [det_results[i_det] for i_det in u_detection]
    
    return matched_tracks, u_strack_pool, u_det_stracks, matched_results, u_matched_results

def matching_iou(strack_pool, det_stracks, threshold, frame_id, det_results=None) :
    dists = matching.iou_distance(strack_pool, det_stracks)
    matches, u_track, u_detection = matching.linear_assignment(dists, threshold)
    
    matched_tracks, matched_results = [], []
    for i_track, i_det in matches :
        track = strack_pool[i_track]
        det_track = det_stracks[i_det]
        if track.state == TrackState.Tracked :
            track.update(det_track, frame_id)
        else :
            track.re_activate(det_track, frame_id, new_id=False)
        
        if det_results is not None :
            track_result = det_results[i_det]
            track_result['id'] = track.track_id
            track_result['bbox'] = track.tlbr
            matched_results.append(track_result)    
        
        matched_tracks.append(track)
    
    u_strack_pool = [strack_pool[i_track] for i_track in u_track]
    u_det_stracks = [det_stracks[i_det] for i_det in u_detection]
    u_matched_results = [det_results[i_det] for i_det in u_detection]
    return matched_tracks, u_strack_pool, u_det_stracks, matched_results, u_matched_results

def set_track_id(stracks, track_info, det_results=None) :
    new_stracks = []
    for idx, track in enumerate(stracks) :
        if track.score < track_info.det_thres :
            continue
        track.activate(track_info.frame_id, track_info)
    
        if det_results is None :
            det_results = []
        else :
            track_result = det_results[idx]
            track_result['id'] = track.track_id
            track_result['bbox'] = track.tlbr
        new_stracks.append(track)
        
    return new_stracks, det_results

def set_removed_tracks(stracks, track_info=None) :
    if track_info is None :
        for strack in stracks :
            strack.mark_removed()
        return stracks, []
    else :
        removed_stracks, unremoved_stracks = [], []
        for strack in stracks :
            if track_info.frame_id - strack.end_frame > track_info.max_time_lost :
                strack.mark_removed()
                removed_stracks.append(strack)
            else :
                unremoved_stracks.append(strack)
        return removed_stracks, unremoved_stracks

def joint_stracks(tlista, tlistb):
    res = set(tlista)
    res = res.union(tlistb)
    res = list(res)
    
    return res

def sub_stracks(tlista, tlistb) :
    stracks = set(tlista)
    stracks = stracks.difference(tlistb)
    stracks = list(stracks)
    return stracks

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb

