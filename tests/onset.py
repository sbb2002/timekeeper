import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# --- 1. 설정 및 가상 오디오 데이터 생성 ---
SAMPLE_RATE = 22050
HOP_LENGTH = 512 # 온셋 감지 결과를 계산할 시간 간격

# 5초 길이의 가상 오디오 신호 생성 (y)
t = np.linspace(0, 5, SAMPLE_RATE * 5, endpoint=False)
y = np.zeros_like(t)

# 0.5초, 1.5초, 3.0초에 "클릭" 소리 시뮬레이션
for click_time in [0.5, 1.5, 3.0]:
    start_idx = int(click_time * SAMPLE_RATE)
    end_idx = start_idx + 200 # 200 샘플 길이의 짧은 충격파
    if end_idx < len(y):
        # 짧은 가우시안 펄스(충격파) 추가
        click_signal = np.exp(-0.5 * (np.linspace(-3, 3, 200)**2)) * 0.5
        y[start_idx:end_idx] += click_signal

# --- 2. 온셋 감지 알고리즘 적용 ---

# 온셋 감지 함수를 계산합니다.
# sr: 샘플 레이트
# hop_length: 온셋 감지 함수를 계산할 프레임 간격
# units='frames'를 사용하면 결과가 프레임 인덱스로 나옵니다.
onset_frames = librosa.onset.onset_detect(
    y=y, 
    sr=SAMPLE_RATE, 
    hop_length=HOP_LENGTH, 
    units='frames' 
)

# 프레임 인덱스를 초(s) 단위로 변환합니다.
onset_times = librosa.frames_to_time(onset_frames, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)

print(f"✅ 감지된 온셋(시작) 시점 (초): {onset_times}")

# --- 3. 시각화 ---

# 1. 오디오 파형 플롯
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=SAMPLE_RATE, ax=plt.gca())

# 2. 감지된 온셋 시점을 세로 선으로 표시
for time in onset_times:
    plt.axvline(x=time, color='r', linestyle='--', linewidth=1, label='Onset')

# 라벨 및 제목 설정
plt.title('Audio Waveform with Onset Detection (Librosa)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# 범례 추가 (중복 표시 방지)
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
plt.legend(unique_labels.values(), unique_labels.keys())

plt.tight_layout()
plt.show()