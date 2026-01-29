from __future__ import annotations

import bisect
import numpy as np
from math import ceil, floor
from collections import deque

np.random.seed(42)

class TimingJudge:
    # 클래스 변수로 한 번만 정의
    THRESHOLDS = [2, 5, 10, 20]     # [ms]
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
    def __init__(self,
                 bpm: int = 60,
                 subnote_denominator: int = 16,
                 samplerate: int = 44100,
                 blocksize: int = 128
                 ):

        # Instance variables
        self.bpm = bpm
        self.subnote_denom = subnote_denominator
        self.sr = samplerate
        self.blk_size = blocksize

        # State: Subnote Counting
        self.n_subnote = 1

        # IVs
        self._update_instance_variables()

        # Buffer: Onset-containing Deque
        dq_maxsize = self._subnote_blk_range_fin - self._subnote_blk_range_init
        self.dq_onset = deque(maxlen=dq_maxsize)

        # History: Results for Timing
        self.hist_results = {
            "delta_t" : [],
            "grade": [],
            "timestamp": [],
            "subnote": []
        }

        # 메모리: 전체 결과에 대한 리포트

    def _update_instance_variables(self):
        # IV 1: Subnote Period
        self._sec_per_note = 60 / self.bpm
        self._subnote_per_note = 4 / self.subnote_denom
        self._subnote_period_unit = self._sec_per_note * self._subnote_per_note
        self._subnote_period = self._subnote_period_unit * self.n_subnote

        # IV 2: Subnote Sample Index
        self._subnote_samp_ix_unit = self.sr * self._subnote_period_unit
        self._subnote_samp_ix = self.sr * self._subnote_period

        # IV 3: Subnote Sample Range
        RANGE_THRESHOLD = 0.3
        boundary = self._subnote_samp_ix_unit * RANGE_THRESHOLD
        print("UNIT:", self._subnote_samp_ix_unit, boundary)

        self._subnote_samp_init = ceil(self._subnote_samp_ix - boundary)
        self._subnote_samp_fin = floor(self._subnote_samp_ix + boundary)
        self._subnote_samp_range = (self._subnote_samp_init, self._subnote_samp_fin)
        print("SAM RANGE:", self._subnote_samp_range)

        # IV 4: Subnote Block Range
        self._subnote_blk_range_init = ceil(self._subnote_samp_range[0] / self.blk_size)
        self._subnote_blk_range_fin = floor(self._subnote_samp_range[1] / self.blk_size)
        self._subnote_blk_range = (self._subnote_blk_range_init, self._subnote_blk_range_fin)
        print("BLK RANGE:", self._subnote_blk_range)

        print("SUBNOTE:", self.n_subnote)
        if hasattr(self, 'dq_onset'):
            print("DQ:", self.dq_onset)
        else:
            print("INITIALIZING")
        print("UPDATED\n")

    def process(self, onset_info: tuple[bool, int]):
        """
        Judging onsets on just-time beat.
        This preceeds sequentially below.

        1. Convert
            `self._subnote_samp_range` -> `_subnote_blk_range`

        2. Collect
            If `onset_info[1]` as `blk_ix` is in `_subnote_blk_range`,
            collect `onset_info[0]` as `onset_bool` at `self.dq_onset`.

        3. Judge
            (1) Get 1st onset if `self.dq_onset` is full.
            (2) Convert 1st onset's `blk_ix` -> 1st onset's `samp_ix`.
                * `samp_ix = (blk_ix = 0.5) * blocksize`
            (3) Measure Δt
                * `Δ(ix) = (1st onset's samp_ix) - (subnote_samp_ix)`
                * `Δt[ms] = Δ(ix) / samplerate * 1000`
                * Judge using class `TimingJudge`
        
        4. Memorize the results.

        5. Update `self.n_subnote += 1`.

        
        :param onset_info: Contains (whether it is onset, its blk_ix).
        :type onset_info: tuple[bool, int]
        """

        # Convert
        subnote_blk_range = self._subnote_blk_range

        # Collect
        curr_blk_ix = onset_info[1]
        is_inrange = subnote_blk_range[0] <= curr_blk_ix <= subnote_blk_range[1]
        # print(subnote_blk_range, curr_blk_ix, is_inrange)

        if is_inrange:
            # print("ADDED:", onset_info)
            self.dq_onset.append(onset_info)

        # Judge
        is_full = len(self.dq_onset) == (self.dq_onset.maxlen - 1)
        if is_full:

            # Get ist onset
            curr_onset_blk_dqix, curr_onset_blk_ix = self._find_first_onset()
            print("DETECTED ONSET:", curr_onset_blk_ix)

            if curr_onset_blk_ix is not None:

                # Convert sample index
                curr_onset_samp_ix = self._convert_bix_into_six(curr_onset_blk_ix)
                curr_onset_t = self._convert_six_into_sec(curr_onset_samp_ix)   # [s]

                # Measure the time differential
                delta_samp_ix = curr_onset_samp_ix - self._subnote_samp_ix
                delta_t = self._convert_six_into_sec(delta_samp_ix, milli_timeunit=True)    # [ms]
                print("DELTA SAMP IX:", delta_samp_ix)

                # Grade delta_t
                grade = TimingJudge.judge(delta_t)

                # Stack history
                self._stack_history(delta_t, grade, curr_onset_t, self.n_subnote)
                print("DEBUG DQ ONSET:", curr_onset_blk_dqix)
                print("DEBUG:", self._debug_process, "\n")

            # Iterate n_subnote and initialize deque
            self.n_subnote += 1
            self._update_instance_variables()
            self.dq_onset.clear()


    def _find_first_onset(self) -> tuple[int, int]:
        """
        Find first onset on current deque.
        
        :returns index_tuple: (Deque index, Total index) for 1st onset.
        """
        return next(
            ((i, v[1]) for i, v in enumerate(self.dq_onset) if v[0]), 
            (None, None))

    def _convert_bix_into_six(self, blk_ix: int) -> int:
        """
        Convert the block index into the sample index.
        
        :param blk_ix: Block index.
        :type blk_ix: int
        :return: Sample index.
        :rtype: int
        """
        return round((blk_ix + 0.5) * self.blk_size)
    
    def _convert_six_into_sec(self, samp_ix: int, milli_timeunit: bool = False) -> float:
        """
        Convert the sample index into mili-second.
        
        :param samp_ix: Sample index.
        :type samp_ix: int
        :param milli_timeunit: Return values uses milli-time unit.
        :type milli_timeunit: bool
        :return: Time second.
        :rtype: float
        """
        adj_msec = 1000 if milli_timeunit else 1
        return samp_ix / self.sr * adj_msec

    def _stack_history(self, 
                       delta_t: float, grade: str, timestamp: float, subnote: int):
        """
        Stack the results.
        
        :param delta_t: Current time[ms] differential between onbeat and 1st onset.
        :type delta_t: float
        :param grade: The grade about `delta_t`.
        :type grade: str
        :param timestamp: Current timestamp for 1st onset.
        :type timestamp: float
        :param subnote: Current subnote index for 1st onset.
        :type subnote: int
        """

        self.hist_results['delta_t'].append(delta_t)
        self.hist_results['grade'].append(grade)
        self.hist_results['timestamp'].append(timestamp)
        self.hist_results['subnote'].append(subnote)

    @property
    def _debug_process(self):
        return {k: v[-1] for k, v in self.hist_results.items()}


#########################
# Onset random sampling #
#########################

def generate_onset_for_onetick(
        jud, tick_len_float,
        tick_period=0.25, bpm=60, sr=44100, blocksize=128):
    
    tick_samp_len = int(60 / bpm * sr * tick_period)    # 0.25s * 44.1kHz = 11025 samples
    tick_length = tick_samp_len / blocksize             # 11025[samps] / 128[samps/blk] = 86.133[blks]
    tick_len_int = floor(tick_length)
    tick_len_float_curr = tick_length % 1

    # Adjusted float point
    tick_len_float += tick_len_float_curr
    if tick_len_float >= 1.0:
        tick_len_int += 1
        tick_len_float -= 1

    one_tick = np.zeros((tick_len_int), dtype=np.int16)  # 86[blks]

    if jud != "first":
        if jud is None:
            is_fast = np.random.rand(1) > 0.5
        else:
            is_fast = True if jud.lower() == "fast" else False

        BEATSIZE = 4
        beatarray = np.ones(BEATSIZE, dtype=np.int16)
        if jud.lower() == "perfect":
            distance = 1
        else:
            boundary = int(tick_length / 50)
            boundary = boundary if boundary > 2 else 2
            distance = np.random.randint(1, boundary) 

        if is_fast:
            one_tick[distance : distance + BEATSIZE] = beatarray
        else:
            one_tick[-(distance + BEATSIZE) : -distance] = beatarray
        # print(one_tick, distance)

    else:
        is_fast = False

    one_tick = one_tick.astype(np.bool)

    return one_tick, is_fast, tick_len_float

def generate_onset_for_onetick2(jud, blk_dist=5):

    PINPOINT = 86
    BEATSIZE = 5

    distance = np.random.randint(1, blk_dist)

    one_tick = np.zeros(PINPOINT, dtype=np.bool)
    beat = np.ones(BEATSIZE, dtype=np.bool)
    
    if jud == "fast":
        one_tick[-(distance + BEATSIZE) : -distance] = beat
    elif jud == "slow":
        one_tick[distance : distance + BEATSIZE] = beat
    elif jud == "perfect":
        one_tick[ : BEATSIZE] = beat
    
    return one_tick, jud, 

def generate_onsets(judge_list):

    onsets = []
    tick_len_float = 0.0
    for jud in judge_list:
        o, _, tick_len_float = generate_onset_for_onetick(jud, tick_len_float=tick_len_float)
        onsets.append(o)
        # print(_)

    return np.concatenate(onsets)

def generate_onset_v3(
        duration=1,
        bpm=60, 
        sr=44100,
        blocksize=128,
        subnote_denom=16
        ):
    
    # Duration length block array
    unit_len = sr / blocksize
    length = duration * unit_len
    soundblk = np.zeros(int(length), dtype=np.int16)

    # Hit on beat
    BEATSIZE = 5

    unit_subnote_len = unit_len * (4 / subnote_denom)
    ixs = int(length // unit_subnote_len)
    for i in range(ixs):
        RND_DIST = np.random.randint(-4, 4)
        ix = unit_subnote_len * i
        ix = round(ix)

        ix = ix + RND_DIST
        soundblk[ix : ix + BEATSIZE] = np.ones(BEATSIZE, dtype=np.int16)

    return soundblk.astype(np.bool)
    
def generate_onset_tuple(onsets):
    return [(o, n) for n, o in enumerate(onsets)]


#######
# Run #
#######

# onset = generate_random_onset(10)
# judlist = ['first', 'perfect', 'slow', 'fast'] * 1
# onset = generate_onsets(judge_list=judlist)
onset = generate_onset_v3(duration=3)
onset = generate_onset_tuple(onset)
# print("ONSET:", onset)

testing = np.array([o[0] for o in onset])
print("ARTIFICIAL ONSET INDEXES:", np.where(testing))
# print(onset)

rc = RhythmChecker(subnote_denominator=16)
for o in onset:
    rc.process(o)

print("====== FINAL RESULTS ======")
print(rc.hist_results)