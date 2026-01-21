import numpy as np
from scipy import signal

class OnsetDetector:
    def __init__(self, sr=44100, hop_length=512, threshold=0.3):
        """
        Spectral Flux 기반 Onset Detector.
        이 검출기는 최소 2개 이상의 연속된 윈도우를 검출해야 정상 작동함!
        
        Parameters:
        -----------
        sr : int
            샘플링 레이트 (Hz)
        hop_length : int
            STFT의 hop size
        threshold : float
            onset 검출 임계값 (0.0 ~ 1.0)
        """
        self.sr = sr
        self.hop_length = hop_length
        self.threshold = threshold
        self.prev_spectrum = None
        self.flux_history = []
        
    def compute_spectral_flux(self, audio_frame):
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
                                noverlap=1024-self.hop_length)
        
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
    
    def detect_onset(self, audio_frame):
        """
        onset 검출
        
        Parameters:
        -----------
        audio_frame : np.ndarray
            4096 샘플의 오디오 데이터
            
        Returns:
        --------
        is_onset : bool
            onset이 검출되었는지 여부
        flux_value : float
            현재 spectral flux 값
        """
        # Spectral flux 계산
        flux = self.compute_spectral_flux(audio_frame)
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


class OnsetPeakPicker:
    def __init__(self, threshold, lookback=1, lookahead=1):
        
        self.threshold = threshold
        self.flux_buffer = []
        self.lookback = lookback
        self.lookahead = lookahead

    def process(self, flux):

        self.flux_buffer.append(flux)
        # print(len(self.flux_buffer))

        # Check enough buffer
        if len(self.flux_buffer) < self.lookahead + self.lookahead + 1:
            return False
        
        # center peak check
        center_ix = self.lookback
        center_val = self.flux_buffer[center_ix]
        # print(center_val >= self.flux_buffer)

        if center_val < self.threshold:
            self.flux_buffer.pop(0)
            # print("Less than threshold")
            return False
    
        # Local maximum
        is_peak = all(center_val >= self.flux_buffer[i] * 0.75
                        for i in range(len(self.flux_buffer))
                        if i != center_ix)
        # print(is_peak)
        
        self.flux_buffer.pop(0)
        return is_peak