from __future__ import annotations
from typing import Tuple

import bisect
import queue
import threading
import numpy as np
from scipy import signal
from collections import deque

class OnsetDetector:
    """
    Onset 검출기 클래스. 
    
    Energy flux를 기반으로 검출하므로 별도의 picking 작업이 필요.

    이 검출기는 최소 2개 이상의 연속된 윈도우를 검출해야 정상 작동함!
    """
    def __init__(self, sr: int=44100, hop_length: int=512, threshold: float=0.3):
        """
        Parameters:
        -----------
        sr : int
            샘플링 레이트 (Hz)
        hop_length : int
            STFT의 hop size
        threshold : float
            Energy flux의 평균값 + N시그마를 설정. (권장: 0.0 ~ 1.0)
        """
        self.sr = sr
        self.hop_length = hop_length
        self.threshold = threshold
        self.prev_spectrum = None
        self.flux_history = []
        
    def detect_onset(self, audio_frame: np.ndarray) -> Tuple[bool, float, float]:
        """
        
        한 윈도우 안에서 onset 후보군을 검출함.

        Parameters:
        -----------
        audio_frame : np.ndarray
            4096 샘플의 오디오 데이터
            
        Returns:
        --------
        `is_onset` : (bool) onset이 검출되었는지 여부.

        `flux` : (float) 현재 spectral flux 값

        `adaptive_threshold` : (float) 현재 적용 중인 flux threshold 값

        """
        # Spectral flux 계산
        flux = self._compute_spectral_flux(audio_frame)
        self.flux_history.append(flux)
        
        # 최근 몇 프레임의 평균으로 adaptive threshold 계산
        min_threshold = 0.01
        window_size = 10
        if len(self.flux_history) < window_size:
            adaptive_threshold = min_threshold
        else:
            recent_fluxes = self.flux_history[-window_size:]
            adaptive_threshold = np.mean(recent_fluxes) + self.threshold * np.std(recent_fluxes)
            adaptive_threshold = max(min_threshold, adaptive_threshold)
        
        # Peak picking: 현재 flux가 threshold보다 크면 onset
        is_onset = flux > adaptive_threshold and flux > 0
        
        return is_onset, flux, adaptive_threshold
    
    def reset(self):
        """
        검출기 상태 초기화
        """
        self.prev_spectrum = None
        self.flux_history = []

    def _compute_spectral_flux(self, audio_frame: np.ndarray) -> np.float32:
        """
        주어진 오디오 프레임에 대해 spectral flux 계산
        
        Parameters:
        -----------
        audio_frame : np.ndarray
            4096 샘플의 오디오 데이터
            
        Returns:
        --------
        flux : float
            현재 프레임의 spectral flux 값
        """
        # STFT 수행
        f, t, Zxx = signal.stft(audio_frame, 
                                fs=self.sr, 
                                nperseg=1024,
                                noverlap=None)
        
        # 각 시간 프레임에 대한 magnitude spectrum
        magnitude = np.abs(Zxx)
        
        # 첫 프레임이면 초기화
        if self.prev_spectrum is None:
            self.prev_spectrum = magnitude[:, 0] if magnitude.shape[1] > 0 else np.zeros(magnitude.shape[0])
            return 0.0
        
        # Spectral Flux 계산 (positive differences only)
        # 각 시간 프레임에 대해 계산
        flux_values = []
        for i in range(magnitude.shape[1]):
            current = magnitude[:, i]
            diff = current - self.prev_spectrum
            # Half-wave rectification: 증가하는 부분만 고려
            diff = np.maximum(0, diff)
            flux = np.sum(diff)
            flux_values.append(flux)
            self.prev_spectrum = current
        
        # 평균 flux 값 반환
        return np.mean(flux_values) if flux_values else 0.0
    

class OnsetPeakPicker:
    """
    Peak onset만 선별하는 클래스.

    한 윈도우에 대해 계산된 flux를 어느 정도 수집 후 
    
    인접한 값을 비교하여 국소 최대값을 peak onset으로 판별함. 
    """
    def __init__(self, threshold: float, lookback: int=1, lookahead: int=1):
        
        self.threshold = threshold
        self.flux_buffer = []
        self.lookback = lookback
        self.lookahead = lookahead

    def pick_peak(self, flux: float) -> bool:
        """
        :param flux: (float) 한 윈도우에 대해 계산된 flux.
        :return: (bool) 해당 윈도우의 peak 여부.
        """
        # Acquire the flux
        self.flux_buffer.append(flux)

        # Check enough buffer
        if self._check_enough_buffer():
            return False
        
        # Center peak check
        center_ix, center_val = self._get_center()
        if center_val < self.threshold:
            self.flux_buffer.pop(0)
            # print("Less than threshold")
            return False
    
        # Local maximum
        is_peak = self._get_peak(center_ix, center_val)
        
        # Cleanup buffer and returning
        self.flux_buffer.pop(0)
        return is_peak

    def _check_enough_buffer(self):
        return len(self.flux_buffer) < self.lookahead + self.lookahead + 1
    
    def _get_center(self):
        center_ix = self.lookback
        center_val = self.flux_buffer[center_ix]
        return center_ix, center_val
    
    def _get_peak(self, center_ix, center_val):
        is_peak = all(center_val >= self.flux_buffer[i] * 0.75
                        for i in range(len(self.flux_buffer))
                        if i != center_ix)
        return is_peak


class OnsetIntervalChecker:
    """
    Picked onsets의 간격을 확인하는 클래스.

    N, N-1 onset의 시간간격이 subnote의 1/2 이하면 같은 박자로 판정하여

    N-1 onset만을 선별하여 줌.
    """
    def __init__(
            self, 
            bpm: int,
            subnote_denominator: int,
            sr: int=44100, 
            blocksize: int = 4096
            ):

        # Settings
        self.sec_per_block = blocksize / sr
        
        sec_per_note = 60 / bpm
        sec_per_subnote = sec_per_note / subnote_denominator
        self.min_interval = sec_per_subnote / 2

        # Onset info
        self.onset_buffer = []
        self.onset_ix = 0

    def check_interval(self, is_peak):
        
        # Acquire
        self.onset_buffer.append(is_peak)

        if len(self.onset_buffer) < 2:
            return False
        
        # Interval check
        all(self.onset_buffer) & (self.sec_per_block > self.min_interval)


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

        w_uni = block_per_subnote * SS_THRESHOLD
        w_fast = w_uni * W_FAST
        w_late = w_uni * W_LATE
        self.W = int(w_fast + w_late)       # 38 blocks per window

        # Memory
        self.current_subnote = 0
        self.current_block = 0
        self.dq = deque(maxlen=self.W + 1)  # windowsize + 여유분

        
    def process(self, onset: bool):
        
        # Test
        print(self.current_block)
        print(self.current_subnote)

        # Collect onsets
        self._collect_onset(onset)
        
        # If current block has subnote timing
        prev_blk_ix = self.current_block - self.W
        curr_blk_ix = self.current_block
        onbeat_blk_ix = self.current_subnote * self.bs

        if prev_blk_ix < onbeat_blk_ix < curr_blk_ix:
            
            # Pick the fastest onset
            first_onset_blk_ix = self._find_first_onset(self.dq)
            if first_onset_blk_ix is not None:
                current_onset_blk_ix = first_onset_blk_ix + self.current_block
                
                # Time differential
                diff_blk = onbeat_blk_ix - current_onset_blk_ix
                diff_sec = diff_blk * self.sb

                # Judge the Quality of Timing as grade
                grade = self._measure_groove(diff_sec)
                print("Grade: ", grade, " // " , diff_sec, "sec")

                # Count current subnote
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

class TimingJudge:
    # 클래스 변수로 한 번만 정의
    THRESHOLDS = [0.002, 0.005, 0.010, 0.020]
    GRADES = ["PERFECT", "GREAT", "GOOD", "BAD", "MISS"]
    
    @classmethod
    def judge(cls, time_diff):
        abs_diff = abs(time_diff)
        index = bisect.bisect_left(cls.THRESHOLDS, abs_diff)
        return cls.GRADES[index]

class RhythmSupervisor:
    """
    Onset 관련한 클래스들을 총괄하는 클래스.

    여기서 추가적으로 해야할건 Onset Index에 대해 타이밍 차이를 산출하기.

    그리고 이제는 latency 최소화를 위해 블록사이즈를 128로 낮춤.
    """
    def __init__(
            self,
            target_buffer: queue.Queue,
            bpm: int,
            subnote_denominator: int,
            samplerate: int=44100, 
            hop_length: int=512, 
            flux_ratio_threshold: float=0.2,
            peak_threshold: float = 0.5,
            lookback: int = 1,
            lookahead: int = 1
            ):
        
        sec_per_note = 60 / bpm
        sec_per_subnote = sec_per_note / subnote_denominator * 4
        self.min_interval = sec_per_subnote

        # Buffer targeting
        self.target_buffer = target_buffer

        # Instances
        self.onset_detector = OnsetDetector(
            sr=samplerate,
            hop_length=hop_length,
            threshold=flux_ratio_threshold
        )
        self.onset_picker = OnsetPeakPicker(
            threshold=peak_threshold,
            lookback=lookback,
            lookahead=lookahead
        )
        self.rhythm_checker = RhythmChecker()

        # Memory
        self.block_frames = 0

    def detect(self, audio_frames: np.ndarray):

        # Onset detection
        _, flux, _ = self.onset_detector.detect_onset(audio_frames)
        is_peak = self.onset_picker.pick_peak(flux)
        
        return is_peak
        
    def process_rhythm(self, socketio_obj=None):

        try:
            # 매 오디오 프레임(blocksize=128)이 들어옴
            audio_frames = self.target_buffer.get_nowait()
            self.block_frames += audio_frames.shape[0]

            # 피크 검출
            is_peak = self.detect(audio_frames)
            self.rhythm_checker.process(is_peak)

            # 타이밍 판정

            print("Peak: ", is_peak)
            # 타이밍 검출

            if socketio_obj is not None:
                socketio_obj.emit(
                    'detection_update', {
                        'onBeat': is_peak
                    }
                )

            # 추후 업데이트 주기를 모니터 주시율로 바꿔야할 듯.
            self.schedule = threading.Timer(
                self.min_interval, self.process_rhythm, args=(socketio_obj,))
            self.schedule.start()
        except queue.Empty:
            print("Queue was empty.")

    def kill_process(self):
        self.schedule.cancel()
        self.schedule = None
        print("Kill process")
