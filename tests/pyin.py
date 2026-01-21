import numpy as np
import librosa
import librosa.display # F0 추정 결과 시각화용 (선택 사항)
import matplotlib.pyplot as plt

# --- 1. 설정 변수 ---
SAMPLE_RATE = 22050  # 샘플링 레이트
DURATION_SEC = 5.0   # 오디오 길이 (5초)
FRAME_SIZE = 2048    # F0 추정에 사용할 윈도우 크기 (FFT 윈도우 크기와 유사)
HOP_LENGTH = 512     # 윈도우 이동 간격 (Callback Block Size와 유사)

# --- 2. 가상 오디오 데이터 생성 (C4 음) ---
f0_hz = 261.63 # C4 주파수
t = np.linspace(0, DURATION_SEC, int(SAMPLE_RATE * DURATION_SEC), endpoint=False)
y = 0.5 * np.sin(2 * np.pi * f0_hz * t) 
y += 0.1 * np.random.randn(y.shape[0]) # 약간의 잡음 추가

# --- 3. F0 추정 및 음계 변환 함수 ---

def frequency_to_midi_note(freq):
    """주파수를 미디 노트 번호로 변환합니다."""
    # 20 Hz 미만은 무음(Noise)으로 간주
    if freq < 20.0:
        return 0 
    # MIDI Note = 12 * log2(F0 / 440 Hz) + 69
    midi_note = 12 * np.log2(freq / 440.0) + 69
    return int(np.round(midi_note))

# --- 4. Librosa YIN 알고리즘 적용 (청크 시뮬레이션) ---

# Librosa의 pitch.yin 함수는 전체 오디오 데이터를 입력으로 받음.
# 실시간 처리를 시뮬레이션하기 위해, pitch 추정 결과를 hop_length 간격으로 나눕니다.

print("🎤 Librosa YIN 알고리즘으로 F0 추정 시작...")

# YIN 알고리즘 실행
f0, voiced_flag, voiced_prob = librosa.pyin(
    y, 
    fmin=librosa.note_to_hz('C2'), 
    fmax=librosa.note_to_hz('C7'), 
    sr=SAMPLE_RATE, 
    frame_length=FRAME_SIZE, 
    hop_length=HOP_LENGTH
)

# YIN 결과는 HOP_LENGTH 간격으로 추정된 F0 값의 배열입니다.
print(f"추정된 F0 결과 배열 길이: {len(f0)}")

# --- 5. 음계 변환 및 출력 ---

print("\n--- 청크별 음계 파악 결과 (시뮬레이션) ---")
for i, freq in enumerate(f0):
    midi_note = frequency_to_midi_note(freq)
    
    # 미디 노트가 0이 아닐 때만 출력 (음성/유효한 음높이가 감지된 경우)
    if midi_note > 0:
        note_name = librosa.midi_to_note(midi_note, cents=True)
        print(f"프레임 {i * HOP_LENGTH // SAMPLE_RATE:.2f}s: F0={freq:.2f} Hz -> {note_name}")
    else:
        print(f"프레임 {i * HOP_LENGTH // SAMPLE_RATE:.2f}s: (No Pitch/Noise)")

# --- 6. 시각화 (선택 사항) ---

plt.figure(figsize=(12, 4))
times = librosa.times_like(f0, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
plt.plot(times, f0, label='F0 by YIN', linewidth=2)
plt.title('Fundamental Frequency (F0) Estimation using Librosa PYIN')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.grid(True)
plt.show()