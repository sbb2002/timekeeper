import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from enum import Enum
from typing import Optional, Tuple


class MeasurementState(Enum):
    WAITING = "waiting"  # 피크 대기 중
    MEASURING = "measuring"  # sustain 측정 중
    COMPLETED = "completed"  # 측정 완료

class GuitarSustainMeter:
    def __init__(
        self, 
        sample_rate: int = 44100,
        buffer_size: int = 4096,
        peak_threshold_db: float = -20.0,
        end_threshold_db: float = -40.0,
        min_sustain_time: float = 0.1,  # 최소 sustain 시간 (초)
        auto_reset_time: float = 0.5  # 자동 리셋 시간 (초)
    ):
        """
        일렉기타 sustain 측정기
        
        Args:
            sample_rate: 샘플링 레이트 (Hz)
            buffer_size: 버퍼 크기 (샘플 수)
            peak_threshold_db: 피크 감지 임계값 (dB)
            end_threshold_db: sustain 종료 임계값 (dB)
            min_sustain_time: 최소 sustain 시간 (초)
            auto_reset_time: 측정 완료 후 자동 리셋까지의 대기 시간 (초)
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.peak_threshold_db = peak_threshold_db
        self.end_threshold_db = end_threshold_db
        self.min_sustain_time = min_sustain_time
        self.auto_reset_time = auto_reset_time
        
        # 선형 스케일로 변환
        self.peak_threshold = self._db_to_linear(peak_threshold_db)
        self.end_threshold = self._db_to_linear(end_threshold_db)
        
        # 상태 변수
        self.state = MeasurementState.WAITING
        self.peak_time = 0  # 피크 감지 시점 (샘플 수)
        self.total_samples_processed = 0  # 처리된 총 샘플 수
        self.sustain_time = None  # 측정된 sustain 시간 (초)
        self.completion_time = 0  # 측정 완료 시점 (샘플 수)
        
        # 레벨 추적
        self.current_rms = 0.0
        self.peak_rms = 0.0
        
        # 측정 결과 저장
        self.last_sustain_time = None
        
    def _db_to_linear(self, db: float) -> float:
        """dB를 선형 스케일로 변환"""
        return 10 ** (db / 20.0)
    
    def _linear_to_db(self, linear: float) -> float:
        """선형 스케일을 dB로 변환"""
        if linear < 1e-10:  # 매우 작은 값 처리
            return -100.0
        return 20 * np.log10(linear)
    
    def _calculate_rms(self, samples: np.ndarray) -> float:
        """RMS (Root Mean Square) 계산"""
        return np.sqrt(np.mean(samples ** 2))
    
    def process_buffer(self, samples: np.ndarray) -> Tuple[Optional[float], str]:
        """
        4096개의 모노 샘플 데이터 처리
        
        Args:
            samples: 4096개의 모노 샘플 데이터 (numpy array, -1.0 ~ 1.0 범위)
        
        Returns:
            (sustain_time, status_message)
            sustain_time: 측정 완료 시 sustain 시간(초), 아니면 None
            status_message: 현재 상태 메시지
        """
        assert len(samples) == self.buffer_size, f"Buffer size must be {self.buffer_size}"
        
        # RMS 계산
        self.current_rms = self._calculate_rms(samples)
        current_db = self._linear_to_db(self.current_rms)
        
        # 총 샘플 수 업데이트
        self.total_samples_processed += self.buffer_size
        
        # 상태 머신
        if self.state == MeasurementState.WAITING:
            # 피크 감지 대기
            if self.current_rms > self.peak_threshold:
                self.state = MeasurementState.MEASURING
                self.peak_time = self.total_samples_processed
                self.peak_rms = self.current_rms
                return None, f"피크 감지! ({current_db:.1f} dB) - 측정 시작"
            else:
                return None, f"대기 중... (현재: {current_db:.1f} dB, 임계값: {self.peak_threshold_db} dB)"
        
        elif self.state == MeasurementState.MEASURING:
            # 피크 레벨 업데이트
            if self.current_rms > self.peak_rms:
                self.peak_rms = self.current_rms
            
            # Sustain 종료 확인
            if self.current_rms < self.end_threshold:
                # Sustain 시간 계산 (초 단위)
                sustain_samples = self.total_samples_processed - self.peak_time
                self.sustain_time = sustain_samples / self.sample_rate
                
                # 최소 sustain 시간 체크
                if self.sustain_time < self.min_sustain_time:
                    # 너무 짧으면 노이즈로 간주하고 리셋
                    self.reset()
                    return None, "측정 리셋 (너무 짧음)"
                
                self.state = MeasurementState.COMPLETED
                self.completion_time = self.total_samples_processed
                self.last_sustain_time = self.sustain_time
                peak_db = self._linear_to_db(self.peak_rms)
                return self.sustain_time, f"측정 완료! Sustain: {self.sustain_time:.3f}초 (피크: {peak_db:.1f} dB)"
            else:
                elapsed = (self.total_samples_processed - self.peak_time) / self.sample_rate
                return None, f"측정 중... {elapsed:.2f}초 (현재: {current_db:.1f} dB)"
        
        elif self.state == MeasurementState.COMPLETED:
            # 자동 리셋: 조용한 상태가 일정 시간 지속되면 다시 대기 상태로
            time_since_completion = (self.total_samples_processed - self.completion_time) / self.sample_rate
            
            if time_since_completion >= self.auto_reset_time:
                # 자동 리셋
                old_sustain = self.sustain_time
                self.reset()
                return None, f"자동 리셋됨 (이전 측정: {old_sustain:.3f}초) - 다음 소리 대기 중"
            else:
                return self.sustain_time, f"측정 완료: {self.sustain_time:.3f}초 ({self.auto_reset_time - time_since_completion:.1f}초 후 자동 리셋)"
        
        return None, "Unknown state"
    
    def reset(self):
        """측정기 리셋"""
        self.state = MeasurementState.WAITING
        self.peak_time = 0
        # total_samples_processed는 리셋하지 않음 (연속 스트림이므로)
        self.sustain_time = None
        self.completion_time = 0
        self.current_rms = 0.0
        self.peak_rms = 0.0
    
    def get_current_level_db(self) -> float:
        """현재 입력 레벨을 dB로 반환"""
        return self._linear_to_db(self.current_rms)
    
    def get_peak_level_db(self) -> float:
        """피크 레벨을 dB로 반환"""
        return self._linear_to_db(self.peak_rms)

# Normalize
def normalize(y):
    print(y.dtype)
    if y.dtype == np.int16:
        return y.astype(np.float32) / 32768.0
    else:
        return y / np.abs(y).max()
    
# Plot
def plot_wave(*ys):
    plt.figure()
    
    for y in ys:
        plt.plot(y, linewidth=0.4, alpha=0.6)
    
    plt.xlabel('Samples')
    plt.ylabel('Amp.')
    # plt.ylim(-1, 1)
    
    plt.show()

# 사용 예시
if __name__ == "__main__":
    # 측정기 초기화
    meter = GuitarSustainMeter(
        sample_rate=44100,
        buffer_size=4096,
        peak_threshold_db=-20.0,
        end_threshold_db=-40.0
    )
    
    # 시뮬레이션: 오디오 인터페이스에서 버퍼를 연속적으로 받는다고 가정
    # 실제로는 pyaudio, sounddevice 등의 콜백에서 이 함수를 호출
    
    print("=== Guitar Sustain Meter ===")
    print("기타 줄을 튕기세요...\n")
    

    # 기타 소리
    wavpath1 = r'tests\Korg-01W-Harmonics1-E5.wav'
    wavpath2 = r'tests\Kawai-K11-CleanGtr-C3.wav'
    # wavpath = r'tests\chirp.wav'
    sr, y1 = wavfile.read(wavpath1)
    sr, y2 = wavfile.read(wavpath2)
    y = np.concatenate([y1, y1, y2])
    y = normalize(y)
    if len(y.shape) >= 2:
        y = y[:, 0]

    signal = y

    # 예시: 간단한 감쇠 신호 생성
    # time_samples = np.arange(0, 2.0, 1/44100)  # 2초 분량
    # # Attack: 빠른 상승, Decay + Sustain: 지수 감쇠
    # signal = np.zeros_like(time_samples)
    # attack_samples = int(0.01 * 44100)  # 10ms attack
    # signal[:attack_samples] = np.linspace(0, 0.8, attack_samples)
    # signal[attack_samples:] = 0.8 * np.exp(-3 * time_samples[:len(time_samples)-attack_samples])
    
    # 버퍼 단위로 처리
    measurement_count = 0
    for i in range(0, len(signal), 4096):
        buffer = signal[i:i+4096]
        if len(buffer) < 4096:
            buffer = np.pad(buffer, (0, 4096 - len(buffer)))
        
        sustain_time, status = meter.process_buffer(buffer)
        
        print(status)
        
        if sustain_time is not None:
            print(f"\n최종 Sustain 시간: {sustain_time:.3f}초")
            print(f"피크 레벨: {meter.get_peak_level_db():.1f} dB")
        
         # 상태 변화가 있을 때만 출력
        if "측정 완료!" in status or "피크 감지!" in status or "자동 리셋" in status:
            print(f"[{i/44100:.2f}s] {status}")
        
        if sustain_time is not None and meter.state == MeasurementState.COMPLETED:
            if meter.last_sustain_time == sustain_time:  # 새로운 측정
                measurement_count += 1
                print(f"  → 측정 #{measurement_count}: {sustain_time:.3f}초\n")

    plot_wave(signal)
    
