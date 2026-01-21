import numpy as np
from scipy import signal

class OnsetDetector:
    def __init__(self, sr=44100, hop_length=512, threshold=0.3):
        """
        Spectral Flux 기반 Onset Detector
        
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
        window_size = 10
        if len(self.flux_history) < window_size:
            adaptive_threshold = 0
        else:
            recent_fluxes = self.flux_history[-window_size:]
            adaptive_threshold = np.mean(recent_fluxes) + self.threshold * np.std(recent_fluxes)
        
        # Peak picking: 현재 flux가 threshold보다 크면 onset
        is_onset = flux > adaptive_threshold and flux > 0
        
        return is_onset, flux
    
    def reset(self):
        """
        검출기 상태 초기화
        """
        self.prev_spectrum = None
        self.flux_history = []


# 사용 예제
if __name__ == "__main__":
    # 테스트용 신호 생성 (4096 샘플)
    sr = 44100
    duration = 4096 / sr
    
    # Onset detector 초기화
    detector = OnsetDetector(sr=sr, threshold=0.5)
    
    # 시뮬레이션: 여러 프레임 처리
    print("Onset Detection 시뮬레이션\n")
    
    for frame_idx in range(20):
        # 프레임 생성 (실제로는 오디오 입력에서 받아옴)
        t = np.linspace(frame_idx * duration, (frame_idx + 1) * duration, 4096)
        
        # 5, 10, 15번째 프레임에서 onset 시뮬레이션 (진폭 증가)
        if frame_idx in [5, 10, 15]:
            audio_frame = 0.5 * np.sin(2 * np.pi * 440 * t)
        else:
            audio_frame = 0.1 * np.sin(2 * np.pi * 440 * t)
        
        # 약간의 노이즈 추가
        audio_frame += 0.01 * np.random.randn(4096)
        
        # Onset 검출
        is_onset, flux = detector.detect_onset(audio_frame)
        
        if is_onset:
            print(f"Frame {frame_idx:3d}: *** ONSET DETECTED *** (flux={flux:.4f})")
        else:
            print(f"Frame {frame_idx:3d}: flux={flux:.4f}")