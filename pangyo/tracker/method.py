# from numba import jit
from .kalman_filter import KalmanFilter
from .basetrack import *
from .byte_tracker import *
from . import matching
import numpy as np
import time

def tracking(detect_result, tracking_info, logger=None) : 
    # Step1 : detect result -> STrack list
    (remain_stracks, remain_results), (second_stracks, second_results), result = create_strack(detect_result, tracking_info)
    if logger is not None :
        logger.debug(f"remain : {len(remain_stracks)}\nsecond : {len(second_stracks)}\nno id : {len(result)}")
        # for track in remain_stracks :
        #     logger.debug(f"{track.score}")
    
    # Step2 : 사람 고유번호를 가지는 STracks 선택
    strack_pool, unconfirmed = track_info_select(tracking_info)
    
    # Step3 : remain 과 strack pool의 fuse & iou 비교
    activate_stracks = [] # 현재 결과에 사온 사람
    lost_stracks = [] # 이전에 등장했던 사람
    
    matched_tracks, strack_pool, remain_stracks, matched_results, remain_results = \
        matching_fuse_iou(strack_pool, remain_stracks, tracking_info.match_thres, tracking_info.frame_id, det_results=remain_results)
    activate_stracks.extend(matched_tracks)
    result.extend(matched_results)
    
    # Step4: 불확실환 결과와 iou 비교
    strack_pool = [strack for strack in strack_pool if strack.state == TrackState.Tracked]
    matched_tracks, strack_pool, second_stracks, matched_results, second_results = \
        matching_iou(strack_pool, second_stracks, 0.5, tracking_info.frame_id, det_results=second_results)
    activate_stracks.extend(matched_tracks)
    for track in strack_pool :
        if track.state == TrackState.Lost :
            track.mark_lost()
    lost_stracks.extend(strack_pool)
    result.extend(matched_results)
    result.extend(second_results)
    
    # Step5: 사라졌던 객체 찾기
    matched_tracks, unconfirmed, remain_stracks, matched_results, remain_results = \
        matching_fuse_iou(unconfirmed, remain_stracks, 0.7, tracking_info.frame_id, det_results=remain_results)
    activate_stracks.extend(matched_tracks)
    result.extend(matched_results)
    
    # Step6: 새로운 아이디 부여
    new_stracks, new_results = set_track_id(remain_stracks, tracking_info, det_results=remain_results)
    activate_stracks.extend(new_stracks)
    result.extend(new_results)
    
    # Step7: track제거
    removed_stracks, lost_stracks = set_removed_tracks(lost_stracks, tracking_info)
    
    # Step8: tracking 정보 최신화
    tracking_info.removed_stracks.extend(removed_stracks)
    activate_stracks, lost_stracks = remove_duplicate_stracks(activate_stracks, lost_stracks) # 같은 사람 제거
    tracking_info.tracked_stracks = activate_stracks
    tracking_info.lost_stracks = lost_stracks
    
    result = sorted(result, key=lambda x : x['order'])
    
    return result, tracking_info

