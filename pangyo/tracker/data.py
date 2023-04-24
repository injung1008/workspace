class TrackingInfo :
    """
    tracking할 비디오 정보
    이전에 나온 사람 정보, frame 번호, 부여된 아이디 수를 가짐
    """
    def __init__(self, 
        frame_id=-1, # frame id
        tracked_stracks=[], # tracking에 사용하는 STrack list
        lost_stracks=[], # tracking에 사용하는 STrack list
        removed_stracks=[], # tracking에 사용하는 STrack list
        count=0, # 부여한 아이디 개수
        max_time_lost = 30, # track을 제거하기 위한 객체가 검출 되지 않은 최소 횟수
        size = (1280, 720), # 원본 이미지 크기(height, width)
        track_thres = 0.6, # 기억할 사람의 최소 점수
        match_thres = 0.9, # 이전 정보와 비교할 점수 기준
        **params # 추가 정보
    ) :
        self.frame_id = frame_id
        self.tracked_stracks = tracked_stracks
        self.lost_stracks = lost_stracks
        self.removed_stracks = removed_stracks
        self.count = count
        self.max_time_lost = max_time_lost
        self.size = size
        self.track_thres = track_thres
        self.match_thres = match_thres
        self.det_thres = self.track_thres + 0.1
        
        for key, value in params.items() :
            setattr(self, key, value)

    def __str__(self) :
        return f"<\n\tframe_id : {self.frame_id}\n\tlenght tracked_strakcs :{len(self.tracked_stracks)}\n\tlength lost_stracks : {len(self.lost_stracks)}\n\tcount : {self.count}\n>"


