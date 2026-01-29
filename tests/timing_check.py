import bisect
import numpy as np
from collections import deque


##############
# Test Class #
##############

class RhythmChecker:
    """
        1) 타이밍 검출 윈도우(W = 1/4 * ss)만큼 오디오 프레임 샘플 수집
        2) 가장 빠른 onset picking
        3) onset ~ window_centroid 비교
        4) 판정 및 업데이트
    """
    def __init__(self, 
                 samplerate: int = 44100,
                 blocksize: int = 128,
                 bpm: int = 60,
                 note_denominator: int = 4,
                 subnote_denominator: int = 4,
                 ):
        
        # Arguments

        # Constants
        SS_THRESHOLD = 0.25
        W_FAST = 0.8
        W_LATE = 1.0

        # Info
        sec_per_note = 60 / bpm
        note_per_subnote = 4 / subnote_denominator
        samples_per_subnote = sec_per_note * note_per_subnote * samplerate
        block_per_subnote = samples_per_subnote / blocksize
        sec_per_block = blocksize / samplerate

        self.ss = samples_per_subnote
        self.bs = block_per_subnote
        self.sb = sec_per_block
        self.bpm = bpm
        self.blocksize = blocksize
        self.note_denom = note_denominator
        self.subnote_denom = subnote_denominator
        self.sec_per_subnote = 4 / subnote_denominator
        self.sr = samplerate

        w_uni = block_per_subnote * SS_THRESHOLD
        w_fast = w_uni * W_FAST
        w_late = w_uni * W_LATE
        self.W = int(w_fast + w_late)       # 38 blocks per window

        # Memory
        self.current_subnote = 0
        self.current_block = 0
        self.dq = deque(maxlen=self.W + 1)  # windowsize + 여유분

        # self.userdata
        self.userdata = {}
        self.userdata['PERFECT'] = 0
        self.userdata['GREAT(FAST)'] = 0
        self.userdata['GREAT(SLOW)'] = 0
        self.userdata['GOOD(FAST)'] = 0
        self.userdata['GOOD(SLOW)'] = 0
        self.userdata['BAD(FAST)'] = 0
        self.userdata['BAD(SLOW)'] = 0
        self.userdata['MISS'] = 0
        self.userdata['t_diff'] = []

    def process(self, onset: bool):
        
        # Collect onsets
        self._collect_onset(onset)
        
        # If current block has subnote timing
        prev_blk_ix = self.current_block - self.W
        curr_blk_ix = self.current_block
        # onbeat_blk_ix = round(self.current_subnote * self.bs)
        onbeat_blk_ix = self.current_subnote * self.sec_per_subnote * self.sr / self.blocksize
        # print(onbeat_blk_ix)

        # print(prev_blk_ix, onbeat_blk_ix, curr_blk_ix)

        if prev_blk_ix < onbeat_blk_ix < curr_blk_ix:
            
            # Pick the fastest onset
            first_onset_blk_ix = self._find_first_onset(self.dq)

            if first_onset_blk_ix is not None:
                
                # # Count current subnote
                # self.current_subnote += 1
                # print("Deque: ", self.dq)
                # print("Found: ", first_onset_blk_ix)
                # print("Onbeat:", onbeat_blk_ix)

                if (self.current_subnote > 4) & (len(self.dq) >= self.W):

                    current_onset_blk_ix = first_onset_blk_ix + self.current_block - self.W
                    
                    # print(self._debug_state)
                    # print(onbeat_blk_ix, current_onset_blk_ix)

                    # Time differential
                    diff_blk = onbeat_blk_ix - current_onset_blk_ix
                    diff_sec = diff_blk * self.sb

                    # Judge the Quality of Timing as grade
                    grade = self._measure_groove(diff_sec)
                    curr_t = onbeat_blk_ix * self.sb
                    print(f"Curr T.: {curr_t:.3f} // ", "Grade: ", grade, " // " , f"{diff_sec * 1000:.1f}ms")

                    self.userdata[grade] += 1
                    self.userdata["t_diff"].append(diff_sec * 1000)

                    self.dq.clear()

        temp_a = self.current_block % self.bs // 0.25
        temp_b = (self.current_block - 1) % self.bs // 0.25

        if temp_a - temp_b < 0:
            # print("CC", self.current_block % self.bs // 0.25)
            self.current_subnote += 1


    def _collect_onset(self, onset: bool):

        try:
            # Append the Onset-Per-Block on deque
            self.dq.append(onset)

            # Count current block
            self.current_block += 1

        except Exception as e:
            print(e)

    def _find_first_onset(self, dq):
        return next((i for i, v in enumerate(dq) if v), None)
    
    def _measure_groove(self, diff_sec):
        return TimingJudge.judge(diff_sec)

    @property
    def _info(self):
        return {
            "BPM": self.bpm,
            "blocksize": self.blocksize,
            "note_denom": self.note_denom,
            "subnote_denom": self.subnote_denom,
        }
    
    @property
    def _info2(self):
        return {
            "samples_per_subnote": self.ss,
            "blocks_per_subnote": self.bs,
            "sec_per_block": self.sb
        }

    @property
    def _debug_state(self):
        """현재 상태 확인용"""
        return {
            'current_block': self.current_block,
            'current_subnote': self.current_subnote,
            'expected_onbeat_blk': round(self.current_subnote * self.bs),
            'deque_size': len(self.dq),
            'deque_range': f"{self.current_block - self.W} ~ {self.current_block}"
        }
    
    @property
    def results(self):
        latency = self.userdata['t_diff']
        l_mean = np.mean(latency)
        l_std = np.std(latency)
        self.userdata.pop('t_diff', None)

        return self.userdata, (l_mean, l_std)

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


################
# Random Input #
################

def generate_random_onset(
        duration=3, tick_period=0.25, bpm=60, sr=44100, blocksize=128,
        threshold=0.5):
    
    onsets = []
    total_ticks = int(duration / tick_period)
    for i in range(total_ticks):
        o, is_fast = generate_onset_for_onetick(tick_period, bpm, sr, blocksize)
        onsets.append(o)

    return np.concatenate(onsets) > threshold

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


#######
# Run #
#######

rc = RhythmChecker(subnote_denominator=16)
# onset = generate_random_onset(10)
judlist = ['fast', 'slow', 'slow', 'slow', 'fast'] * 100
onset = generate_onsets(judge_list=judlist)

print(onset)

print(rc._info)
print(rc._info2)

for o in onset:
    rc.process(o)
    # print(rc._debug_state)

print(rc.results)