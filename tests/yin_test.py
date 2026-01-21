import numpy as np
import matplotlib.pyplot as plt

# Generate the sound wave
def generate_wave(duration, freq=440, sr=44100):
    
    SAMPLE_LEN = duration * sr
    DECAYRATE = 3.2
    
    VOLUME = 0.3
    AMP_0 = 1.00
    AMP_1 = 0.05
    AMP_2 = 0.01

    # Amplitude
    max_amp = AMP_0 + AMP_1 + AMP_2
    amp0 = VOLUME * AMP_0 / max_amp
    amp1 = VOLUME * AMP_1 / max_amp
    amp2 = VOLUME * AMP_2 / max_amp
    
    # Phase
    t = np.linspace(0, duration, SAMPLE_LEN)
    phase = 2 * np.pi * t
    
    # f0, harmonics
    y0 = amp0 * np.sin(1 * freq * phase)
    y1 = amp1 * np.sin(2 * freq * phase)
    y2 = amp2 * np.sin(3 * freq * phase)
    
    y1 = amp1 * np.sin(2 * freq * give_delay(phase, 0.001))
    y2 = amp2 * np.sin(3 * freq * give_delay(phase, 0.002))
    
    # 5 degrees
    freq_5deg = freq * np.power(2, 7/12)
    y05 = amp1 * np.sin(freq_5deg * phase)
    
    # y05 = amp0 * np.sin(freq_5deg * give_delay(phase, 0.001))
    
    # Decay
    wave = y0 + y1 + y2 + y05
    wave = np.exp(-1 * DECAYRATE * t) * wave
    return wave

def give_delay(y, delay, sr=44100):
    
    delay = int(sr * delay)
    
    y = np.roll(y, delay)
    y[: delay] = 0
    return y

# Plot
def plot_wave(*ys):
    plt.figure()
    
    for y in ys:
        plt.plot(y, linewidth=0.4, alpha=0.6)
    
    plt.xlabel('Samples')
    plt.ylabel('Amp.')
    # plt.ylim(-1, 1)
    
    plt.show()

# Yin
def ultra_fast_yin(data, fs, threshold=0.15):
    # 1. 1차원 보장 및 가벼운 전처리
    data = np.asarray(data).flatten()
    data = data / np.max(np.abs(data))  # Waveform normalization
    
    N = len(data)
    tau_max = N // 2
    
    # 2. Difference Function 계산 (루프 없는 버전)
    # d(tau) = sum(x[t]^2) + sum(x[t+tau]^2) - 2*sum(x[t]*x[t+tau])
    
    # 에너지 항 계산
    w = N - tau_max
    x_squared = data**2
    energy_cumsum = np.concatenate([[0], np.cumsum(x_squared)])
    
    # 처음 윈도우의 에너지
    pre_energy = np.sum(x_squared[:w])
    # 각 tau에 대한 에너지를 cumsum으로 빠르게 계산
    # (약간의 근사치를 사용하면 더 빨라지지만, 여기서는 정확도를 위해 rfft 사용)
    
    # FFT를 이용한 상호상관(Cross-correlation)
    n_fft = 2**int(np.ceil(np.log2(2 * N)))
    f_data = np.fft.rfft(data, n=n_fft)
    corr = np.fft.irfft(f_data * np.conj(f_data))
    corr = corr[:tau_max]
    
    # Difference function (루프 없이 벡터 연산)
    # d(tau) = (energy at 0) + (energy at tau) - 2 * corr
    # 실시간 짧은 구간에서는 energy at tau를 energy at 0로 근사 가능
    # cumulative_energy = np.sum(x_squared) 
    fixed_energy = energy_cumsum[w] - energy_cumsum[0]
    shifted_energy = energy_cumsum[w + np.arange(tau_max)] - energy_cumsum[np.arange(tau_max)]
    
    diff = fixed_energy + shifted_energy - 2 * corr
    diff[0] = 1
    # diff = 2 * cumulative_energy - 2 * corr
    # diff[0] = 1 # 0나누기 방지
    
    # diff = np.zeros(tau_max)
    # current_energy = pre_energy
    # for tau in range(tau_max):
    #     if tau > 0:
    #         current_energy = current_energy - x_squared[tau-1] + x_squared[w+tau-1]
    #     diff[tau] = pre_energy + current_energy - 2 * corr[tau]


    # 3. CMNDF (루프 없이 계산)
    # running_sum[tau] = sum(diff[1:tau+1])
    running_sum = np.cumsum(diff[1:])
    tau_indices = np.arange(1, tau_max)
    cmndf = np.ones(tau_max)
    cmndf[1:] = diff[1:] / ((1 / tau_indices) * running_sum)

    # 4. 피크 탐색 (벡터화된 조건 검색)
    possible = np.where(cmndf < threshold)[0]
    if len(possible) > 0:
        # 첫 번째로 임계값을 넘는 구간의 첫 골짜기 찾기
        tau = possible[0]
        # 국소 최솟값 정밀화
        actual_tau = np.argmin(cmndf[tau:min(tau+20, tau_max)]) + tau
        
        # 포물선 보간 (생략 가능하나 정확도를 위해 유지)
        if 0 < actual_tau < tau_max - 1:
            y0, y1, y2 = cmndf[actual_tau-1:actual_tau+2]
            denom = 2*y1 - y0 - y2
            p = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-10 else 0
            return fs / (actual_tau + p)
        return fs / actual_tau
        
    return 0

def ultra_fast_yin2(data, fs, threshold=0.15):
    data = np.asarray(data, dtype=np.float64).flatten()
    # 1. 분석 범위를 명확히 설정 (YIN은 보통 윈도우 크기를 고정함)
    W = 1024  # 분석할 기본 윈도우 크기
    tau_max = len(data) - W - 1 # tau가 이동할 수 있는 최대 범위
    
    if tau_max <= 0: return 0

    print("Data: ", data. data.shape)
    print("Tau: ", tau_max)

    # 2. Difference Function (정석 루프 - 우선 정확도 확인용)
    diff = np.zeros(tau_max)
    for tau in range(1, tau_max):
        # x[j]와 x[j+tau]의 차이의 제곱합
        tmp = data[:W] - data[tau:tau+W]
        diff[tau] = np.sum(tmp**2)
    
    print("Diff: ", diff, diff.shape)
    
    # 3. CMNDF
    cmndf = np.ones(tau_max)
    running_sum = 0
    for tau in range(1, tau_max):
        running_sum += diff[tau]
        cmndf[tau] = diff[tau] / ((1/tau) * running_sum)

    print("Cmndf: ", cmndf, cmndf.shape)
    
    plot_wave(diff, cmndf)

    # 4. 피크 탐색
    possible = np.where(cmndf < threshold)[0]
    if len(possible) > 0:
        # 첫 번째 골짜기 찾기 (첫 번째 임계값 통과 후의 로컬 미니멈)
        for i in range(possible[0], tau_max - 1):
            if cmndf[i] < cmndf[i+1]: # 골짜기 바닥 확인
                actual_tau = i
                break
        else:
            actual_tau = possible[0]

        # 포물선 보간
        if 0 < actual_tau < tau_max - 1:
            y0, y1, y2 = cmndf[actual_tau-1], cmndf[actual_tau], cmndf[actual_tau+1]
            denom = 2*y1 - y0 - y2
            p = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-10 else 0
            return fs / (actual_tau + p)
        return fs / actual_tau
        
    return 0


# LPF
class LowPassFilter(object):
    def __init__(self, cut_off_freqency, ts):
    	# cut_off_freqency: 차단 주파수
        # ts: 주기
        
        self.ts = ts
        self.cut_off_freqency = cut_off_freqency
        self.tau = self.get_tau()

        self.prev_data = 0.
        
    def get_tau(self):
        return 1 / (2 * np.pi * self.cut_off_freqency)

    def filter(self, data):
        val = (self.ts * data + self.tau * self.prev_data) / (self.tau + self.ts)
        self.prev_data = val
        return val



if __name__ == "__main__":
    WINDOW = 1024
    SAMPLERATE = 48000
    YIN_THRESHOLD = 0.20
    
    wave1 = generate_wave(1, freq=110, sr=SAMPLERATE)
    wave2 = generate_wave(1, freq=400, sr=SAMPLERATE)
    wave_raw = np.concatenate(
        [wave1[:int(len(wave1)/2)], wave2])
    
    # Apply LPF
    lpf = LowPassFilter(200, 1/SAMPLERATE)
    # wave = lpf.filter(wave_raw)
    wave = wave_raw
    
    yin1 = ultra_fast_yin2(
        # wave[100 : 100 + WINDOW * 4], 
        wave[:WINDOW * 4], 
        SAMPLERATE, YIN_THRESHOLD)
    yin2 = ultra_fast_yin2(
        wave[len(wave)//2 + 200 : len(wave)//2 + 200 + WINDOW * 4], 
        SAMPLERATE, YIN_THRESHOLD)
    
    print(yin1, yin2)  # A2(110Hz)일 때 window >= 4WINDOW는 되어야 인식 OK
    
    plot_wave(wave, wave_raw)
    
    
"""_summary_
기타의 공명음을 시뮬레이션했다.
이 음은 1도, 5도, 다음 1도, 다다음 1도를 섞은 것이다.
이 음으로 Yin 알고리즘을 돌려보니 A2(110Hz)가 약 116Hz,
E3(165Hz)가 약 174Hz로 예측했다.
꽤 정확했으나 평균율(반음간격 주파수 2**(1/12)배) 특성상
이건 +1반음만큼 오차가 있는 것이다.

yin알고리즘 2로 수정했더니 잘 맞춘다.
yin알고리즘의 Difference function 정밀도 문제였다고 한다.
"""