import bisect
import numpy as np
from collections import deque


class TimingJudge:
    # 클래스 변수로 한 번만 정의
    THRESHOLDS = [0.002, 0.005, 0.010, 0.020]
    GRADES = ["PERFECT", "GREAT", "GOOD", "BAD", "MISS"]
    
    @classmethod
    def judge(cls, time_diff):
        abs_diff = abs(time_diff)
        index = bisect.bisect_left(cls.THRESHOLDS, abs_diff)
        if (index != 0) & (index != 4):
            timing = "(SLOW)" if np.sign(time_diff) ==1 else "(FAST)"
        else:
            timing = ""

        return f"{cls.GRADES[index]}{timing}"


class RhythmChecker:
    """리듬 타이밍 체크 클래스
    
    동작 방식:
    1) 윈도우 크기만큼 onset 수집
    2) 정확한 타이밍이 윈도우에 들어오면 가장 빠른 onset 찾기
    3) 타이밍 차이 계산 및 판정
    """
    
    def __init__(self, 
                 samplerate: int = 44100,
                 blocksize: int = 128,
                 bpm: int = 60,
                 note_denominator: int = 4,
                 subnote_denominator: int = 16):
        
        # 시간 계산
        sec_per_beat = 60.0 / bpm
        sec_per_note = sec_per_beat * (4.0 / note_denominator)
        self.sec_per_subnote = sec_per_note / (note_denominator / subnote_denominator)
        self.sec_per_block = blocksize / samplerate
        
        # 윈도우 크기 계산 (subnote의 25% 범위)
        blocks_per_subnote = self.sec_per_subnote / self.sec_per_block
        window_blocks = blocks_per_subnote * 0.25
        self.W = int(window_blocks * 1.8)  # fast(0.8) + late(1.0) = 1.8
        
        # 상태 변수
        self.current_block = 0
        self.current_subnote = 0
        self.dq = deque(maxlen=self.W + 1)
        
        # 결과 저장
        self._init_userdata()
        
        # 디버그 정보
        self.bpm = bpm
        self.blocksize = blocksize
    
    def _init_userdata(self):
        """결과 저장소 초기화"""
        grades = ['PERFECT', 'GREAT(FAST)', 'GREAT(SLOW)', 
                  'GOOD(FAST)', 'GOOD(SLOW)', 'BAD(FAST)', 
                  'BAD(SLOW)', 'MISS']
        self.userdata = {grade: 0 for grade in grades}
        self.userdata['t_diff'] = []
    
    def process(self, onset: bool):
        """onset 처리 및 타이밍 판정"""
        self.dq.append(onset)
        self.current_block += 1
        
        # 현재 subnote 번호 업데이트 (누적 오차 방지)
        self._update_subnote_counter()
        
        # 타이밍 체크 및 판정
        if self._is_timing_window():
            self._judge_timing()
    
    def _update_subnote_counter(self):
        """현재 시간 기준으로 subnote 번호 업데이트"""
        current_time = self.current_block * self.sec_per_block
        expected_subnote = int(current_time / self.sec_per_subnote)
        if expected_subnote > self.current_subnote:
            self.current_subnote = expected_subnote
    
    def _is_timing_window(self) -> bool:
        """정확한 타이밍이 윈도우 범위 안에 있는지 확인"""
        expected_time = self.current_subnote * self.sec_per_subnote
        onbeat_block = round(expected_time / self.sec_per_block)
        
        window_start = self.current_block - self.W
        window_end = self.current_block
        
        return window_start < onbeat_block < window_end
    
    def _judge_timing(self):
        """타이밍 판정 수행"""
        # 최초 onset 찾기
        first_onset_idx = self._find_first_onset()
        if first_onset_idx is None:
            return
        
        # 워밍업 기간 스킵
        if self.current_subnote <= 4 or len(self.dq) < self.W:
            return
        
        # 시간 차이 계산
        onset_block = self.current_block - self.W + first_onset_idx
        onset_time = onset_block * self.sec_per_block
        expected_time = self.current_subnote * self.sec_per_subnote
        diff_sec = expected_time - onset_time
        
        # 판정 및 기록
        grade = TimingJudge.judge(diff_sec)
        self._record_result(grade, diff_sec, expected_time)
        
        # 윈도우 초기화
        self.dq.clear()
    
    def _find_first_onset(self):
        """deque에서 첫 번째 onset의 인덱스 찾기"""
        return next((i for i, v in enumerate(self.dq) if v), None)
    
    def _record_result(self, grade: str, diff_sec: float, time_sec: float):
        """결과 기록 및 출력"""
        self.userdata[grade] += 1
        self.userdata['t_diff'].append(diff_sec * 1000)
        print(f"Time: {time_sec:.3f}s // Grade: {grade} // Diff: {diff_sec * 1000:.1f}ms")
    
    @property
    def results(self):
        """최종 결과 반환"""
        latency = self.userdata['t_diff']
        if latency:
            l_mean = np.mean(latency)
            l_std = np.std(latency)
        else:
            l_mean, l_std = 0.0, 0.0
        
        results = self.userdata.copy()
        results.pop('t_diff', None)
        
        return results, (l_mean, l_std)
    
    def _debug_state(self):
        """디버깅용 상태 정보"""
        expected_time = self.current_subnote * self.sec_per_subnote
        return {
            'current_block': self.current_block,
            'current_time': f"{self.current_block * self.sec_per_block:.3f}s",
            'current_subnote': self.current_subnote,
            'expected_time': f"{expected_time:.3f}s",
            'deque_size': len(self.dq),
        }
    


def generate_onset_for_onetick(
        jud=None,
        tick_period=0.25, bpm=60, sr=44100, blocksize=128,
        err_ratio=0.2):
    
    tick_length = int(sr / blocksize * tick_period)
    one_tick = np.zeros((tick_length))

    if jud is None:
        is_fast = np.random.rand(1) > 0.5
    else:
        is_fast = True if jud.lower() == "fast" else False

    beatsize = int(err_ratio * one_tick.shape[0])
    beatarray = np.random.rand(beatsize)

    if is_fast:
        one_tick[: beatsize] = beatarray
    else:
        one_tick[-beatsize: ] = beatarray

    # print(one_tick[:5], one_tick[-5:])

    return one_tick, is_fast

def generate_onsets(judge_list, threshold=0.5):

    onsets = []
    for jud in judge_list:
        o, _ = generate_onset_for_onetick(jud, err_ratio=0.05)
        onsets.append(o)
        # print(_)

    return np.concatenate(onsets) > threshold



rc = RhythmChecker(subnote_denominator=16)
# onset = generate_random_onset(10)
judlist = ['fast', 'slow', 'slow', 'slow', 'fast'] * 100
onset = generate_onsets(judge_list=judlist)

print(onset)

for o in onset:
    rc.process(o)
    # print(rc._debug_state)

print(rc.results)