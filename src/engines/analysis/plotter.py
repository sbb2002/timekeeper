import queue
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import stft, decimate
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from common.handler import PrintHandler


# Energy
def get_energy(buffer):
    return np.sqrt(buffer ** 2)

# Onset Detection
def get_odf(zxx):
    ## 1) STFT
    # _, _, zxx = stft(energy.reshape((-1,)))
    
    ## 2) Magnitude & phase
    mag = np.abs(zxx)
    phase = np.angle(zxx)
    # self.prtwl("STFT MAG:", mag.shape)
    
    ## 3) Onset Detecting by mag
    prev_mag = mag[:, :-1]
    curr_mag = mag[:, 1:]
    
    diff_mag = curr_mag - prev_mag
    halfwave_rectified = np.maximum(0, diff_mag)
    spectral_flux = np.sum(halfwave_rectified, axis=0)
    
    ## 4) Onset Detecting by phase
    prev_phase = phase[:, :-1]
    curr_phase = phase[:, 1:]
    
    delta_phase = curr_phase - prev_phase
    unwrapped_delta_phase = np.unwrap(delta_phase, axis=1)
    weigthed_phase_deviation = prev_mag * np.abs(unwrapped_delta_phase)
    
    odf_phase = np.sum(weigthed_phase_deviation, axis=0)
    
    ## 5) Combined Onset Detecting Function
    odf = spectral_flux + odf_phase
    
    return odf

def set_threshold(odf, factor=1.5, offset=0.1, window_size=20):
    
    # Moving average
    window_size = max(min(odf.shape[0] - 2, window_size), 12)
    # local_mean = np.convolve(odf, np.ones(window_size)/window_size, mode='same')
    local_mean = uniform_filter1d(odf, size=window_size)
    
    # Set threshold
    threshold = local_mean * factor + offset
    
    return threshold

def detect_onset(odf, threshold, n_peak_samples=3):
    # Peak picking
    max_odf = maximum_filter1d(odf, size=n_peak_samples)
    is_peak = (odf == max_odf)
    
    # And over threshold
    candidate_onsets_mask = is_peak & (odf > threshold)
    
    # Get onset index and strength
    onset_segms = np.where(candidate_onsets_mask)[0]
    onset_strengths = odf[onset_segms]
    
    return onset_segms, onset_strengths

def convert_segms_into_frames(segms, total_segms, blocksize):
    if segms.size == 0:
        return segms
    
    frames_per_total_segms = blocksize
    return np.round(
        segms / total_segms * frames_per_total_segms
        ).astype(int)

def merge_onsets_by_strength(onset_frames, onset_strengths, sr=44100):
    
    # Set frame distance
    FRAME_THRESHOLD = get_distance_frames_for_temporal_resolution(sr=sr)
    
    if onset_frames.size == 0:
        return np.array([])
    
    # 1) Groupping
    is_new_group_start = np.zeros_like(onset_frames, dtype=bool)
    is_new_group_start[0] = True
    
    current_group_start_frame = onset_frames[0]
    for i in range(1, onset_frames.size):
        if onset_frames[i] - current_group_start_frame >= FRAME_THRESHOLD:
            is_new_group_start[i] = True
            current_group_start_frame = onset_frames[i]
    
    group_ids = np.cumsum(is_new_group_start) - 1
    
    # 2) Max. strength by group
    group_indices = np.split(
        np.arange(onset_frames.size), 
        np.where(np.diff(group_ids))[0] + 1)
    
    final_onsets = []
    
    for indices in group_indices:
        group_strengths = onset_strengths[indices]
        max_strength_local_ix = np.argmax(group_strengths)
        representative_onset_ix = indices[max_strength_local_ix]
        
        final_onsets.append(onset_frames[representative_onset_ix])

    return np.array(final_onsets)
    
def validate_first_onset_connecting_last_onset(current_onset_frames, last_onset_frames, total_frames, sr=44100):
    
    # Set frame distance
    FRAME_THRESHOLD = get_distance_frames_for_temporal_resolution(sr=sr)
    
    # If current onset frames exist
    if current_onset_frames.size > 0:
        # If last onset frames exist
        if last_onset_frames.size > 0:
            # Measure distance from last to current onset
            prev_onset = last_onset_frames[-1]
            curr_onset = current_onset_frames[0] + total_frames
            distance = curr_onset - prev_onset
            # If farther than threshold, ignore the 1st element of current onsets
            if distance < FRAME_THRESHOLD:
                current_onset_frames = current_onset_frames[1:]
        # Memory this onsets for next
        last_total_onset_frames = current_onset_frames + total_frames
    
    # Else returns raw
    else:
        current_onset_frames = current_onset_frames
        last_total_onset_frames = last_onset_frames
    
    return current_onset_frames, last_total_onset_frames

def get_distance_frames_for_temporal_resolution(sr=44100):
    TEMPORAL_RESOLUTION = 20    # ms
    return round(sr * TEMPORAL_RESOLUTION / 1000)

# Pitch Detection
def get_refined_pitch(f, zxx, sr):
    ## 1) Get magnitude
    mag = np.log1p(np.abs(zxx))
    if mag.max() < 0.05:
        return 0    # if silent
    
    ## 2) HPS(Harmonic Product Spectrum)
    hps = np.copy(mag)
    for i in range(2, 5):       # 2nd, 3rd harmonics is considering
        downsampled = mag[::i]
        hps[:len(downsampled)] *= downsampled

    ## 3) Rough Peak
    max_search_ix = np.argmin(np.abs(f - 1000))
    try:
        peak_ix = np.argmax(hps[:max_search_ix])
        df = f[1] - f[0]
        
    except:
        peak_ix = np.argmax(hps)
        df = 1
        
    ## 4) Parabolic Interpolation
    if 0 < peak_ix < len(hps) - 1:
        alpha = hps[peak_ix - 1]
        beta = hps[peak_ix]
        gamma = hps[peak_ix + 1]
        
        denom = 2 * beta - (alpha + gamma)
        if abs(denom) >= 1e-10:
            p = 0.5 * (alpha - gamma) / denom
            pitch = (peak_ix + p) * df
            # refined_ix = peak_ix + p
            # try:
            #     sampler = sr / (2 * (len(f) - 1))
            # except ZeroDivisionError:
            #     print("ZeroDivision is occured.", len(f))
            #     sampler = 1
            # pitch = refined_ix * sampler
            
        else:
            pitch = peak_ix * df
    else:
        pitch = peak_ix * df
        
    return pitch

def get_ptich_using_yin(data, sr, threshold=0.1):
    ## 1) Difference Function
    N = len(data)
    tau_max = N // 2
    diff = np.zeros(tau_max)
    
    for tau in range(1, tau_max):
        delta = data[tau:] - data[: -tau]
        diff[tau] = np.sum(delta ** 2)
        
    ## 2) Cumulative Mean Normalized Difference Function
    cmndf = np.zeros(tau_max)
    cmndf[0] = 1
    running_sum = 0
    for tau in range(1, tau_max):
        running_sum += diff[tau]
        cmndf[tau] = diff[tau] / ((1 / tau) * running_sum)
        
    ## 3) Absolute Threshold
    possible_taus = np.where(cmndf < threshold)[0]
    if len(possible_taus) > 0:
        tau = possible_taus[0]
        while tau + 1 < tau_max and cmndf[tau + 1] < cmndf[tau]:
            tau += 1
    else:
        tau = np.argmin(cmndf)
        
    ## 4) Parabolic Interpolation & Freq conversion
    if tau > 0 and tau < tau_max - 1:
        y0, y1, y2 = cmndf[tau-1], cmndf[tau], cmndf[tau+1]
        p = 0.5 * (y0 - y2) / (y0 + y2 - 2 * y1)
        refined_tau = tau + p
        pitch = sr / refined_tau
    else:
        pitch = sr / tau if tau != 0 else 0
        
    return pitch if 50 < pitch < 2000 else 0

def fast_yin(data, fs, threshold=0.15):
    
    data = np.asarray(data).flatten()
    
    N = len(data)
    tau_max = N // 2
    
    # 1. Difference Function을 FFT로 최적화 (핵심!)
    # d(tau) = sum(x[t] - x[t+tau])^2 계산의 고속 버전
    # 에너지를 미리 계산
    energy = np.sum(data**2)
    # FFT 기반 자기상관(Autocorrelation) 계산
    # rfft를 사용하면 복소수 연산이 빨라집니다.
    n_fft = 2**int(np.ceil(np.log2(2 * N - 1)))
    data_fft = np.fft.rfft(data, n=n_fft)
    res = np.fft.irfft(data_fft * np.conj(data_fft))[:tau_max]
    
    # YIN의 difference function 유도식: d(tau) = e(0) + e(tau) - 2*r(tau)
    # 편의상 실시간 1024 샘플에서는 아래의 근사식을 사용하거나 numpy 벡터로 처리
    tau = np.arange(tau_max)
    # 루프 대신 벡터 연산으로 차이 함수 계산
    diff = np.zeros(tau_max)
    for t in range(1, tau_max): # 이 루프는 tau_max(512)만큼만 돌아서 훨씬 빠름
        # 더 극단적인 최적화는 아래 루프도 넘파이 슬라이싱으로 대체 가능
        diff[t] = energy + np.sum(data[:N-t]**2) - 2 * res[t]

    # 2. CMNDF 계산 (벡터화)
    # running_sum을 cumsum으로 한 번에 처리
    diff[0] = 1
    running_sum = np.cumsum(diff[1:])
    idx = np.arange(1, tau_max)
    cmndf = np.zeros(tau_max)
    cmndf[0] = 1
    cmndf[1:] = diff[1:] / ((1 / idx) * running_sum)

    # 3. 피치 추출 (기존과 동일하되 인덱스 제한)
    possible_taus = np.where(cmndf < threshold)[0]
    if len(possible_taus) > 0:
        tau = possible_taus[0]
        # 4. 정밀 보정 (Parabolic Interpolation)
        if 0 < tau < tau_max - 1:
            y0, y1, y2 = cmndf[tau-1], cmndf[tau], cmndf[tau+1]
            denom = 2 * y1 - y0 - y2
            if abs(denom) > 1e-10:
                p = 0.5 * (y0 - y2) / denom
                pitch = fs / (tau + p)
            else:
                pitch = fs / tau
        else:
            pitch = fs / tau
    else:
        pitch = 0 # 피치 못 찾음

    return pitch if 50 < pitch < 1200 else 0 # 기타/목소리 대역 필터링

def ultra_fast_yin(data, fs, threshold=0.15):
    # 1. 1차원 보장 및 가벼운 전처리
    data = np.asarray(data).flatten()
    N = len(data)
    tau_max = N // 2
    
    # 2. Difference Function 계산 (루프 없는 버전)
    # d(tau) = sum(x[t]^2) + sum(x[t+tau]^2) - 2*sum(x[t]*x[t+tau])
    
    # 에너지 항 계산
    w = N - tau_max
    x_squared = data**2
    # 처음 윈도우의 에너지
    power = np.sum(x_squared[:w])
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
    cumulative_energy = np.sum(x_squared) 
    diff = 2 * cumulative_energy - 2 * corr
    diff[0] = 1 # 0나누기 방지

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

# Matplotlib plotter
class MatplotlibPlotter(PrintHandler):
    def __init__(self, data, samplerate, blocksize, duration=5):
        
        # Arguments
        self.data = data
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.duration = duration
        
        # Plot settings
        self.xlim = int(samplerate * duration)
        self.xaxis = np.arange(self.xlim) / samplerate
        self.plot_array = np.zeros(self.xlim, dtype=np.float32)
        # self.dots_array = np.zeros(self.xlim, dtype=np.float32)
        
        # Debug
        self.total_frames = np.int64(0)
        self.last_onsets = np.array([])

        self.initialize()
        
        self.ani = FuncAnimation(
            self.fig,
            self.update,
            interval=10,
            blit=True,
            cache_frame_data=False
        )
        plt.show()
        
    def update(self, frame):
        
        buffer = []
        while True:
            try:
                # print("Get data")
                buffer.append(self.data.get_nowait())
            except queue.Empty:
                # self.prtwl("Queue was empty!")
                break
        
        if buffer:
            # Collect all data in buffer
            data = np.concatenate(buffer, axis=0).flatten()
            
            # Energy and STFT
            energy = get_energy(data)
            # self.prtwl("ENERGY:", energy.shape)
            
            f, t, zxx = stft(energy, nperseg=4096)
            # if f[0] > 0:
            # self.prtwl("STFT:", f.shape, zxx.shape)
                
            # Onset Detection
            odf = get_odf(zxx)
            threshold = set_threshold(odf, factor=1.2, offset=0.1, window_size=20)
            
            try:
                # Onset Detection
                onset_segms, onset_strengths = detect_onset(odf, threshold)
                onset_frames = convert_segms_into_frames(
                    onset_segms,
                    total_segms=odf.shape[0],
                    blocksize=self.blocksize
                    )
                
                # Onset filtering in this block
                final_onsets = merge_onsets_by_strength(
                    onset_frames, onset_strengths,
                    sr=self.samplerate,
                )
                
                # Onset filtering between last and current block
                current_onset_frames, self.last_onsets = validate_first_onset_connecting_last_onset(
                    current_onset_frames=final_onsets,
                    last_onset_frames=self.last_onsets,
                    total_frames=self.blocksize,
                    sr=self.samplerate
                )
                
                if current_onset_frames.size > 0:
                    self.prtwl("ONSET:", current_onset_frames)
                    
                    # If onset, Pitch Detection
                    DOWNSAMPLING_FACTOR = 2
                    downsampled_data = decimate(data, DOWNSAMPLING_FACTOR)
                    downsampled_sr = self.samplerate / DOWNSAMPLING_FACTOR
                    pitch = ultra_fast_yin(
                        downsampled_data, downsampled_sr,
                        threshold=0.7)
                    if pitch > 0:
                        self.prtwl("PITCH:", pitch)

            # If CPU too much overloaded, this block will be ignored.
            except ValueError:
                self.prtwl("Too much shrinken shape. It seems CPU was overloaded. Discarding onset detection for this frame.")
            
            # For other exceptions
            except Exception as e:
                self.prtwl("Unexpected exception:", e)

            # Anyway, counting this num of frames in the end of callback
            finally:
                self.total_frames += data.shape[0]
                

            # Update plot array
            try:
                self.plot_array[: -len(data)] = self.plot_array[len(data):]
                self.plot_array[-len(data):] = data.reshape((-1,))
                
            except ValueError as e:
                self.prtwl("ValueError in plot update:", str(e))
                
            # Update line
            self.line.set_ydata(self.plot_array)
            # self.texts.set_text(f"ONSET: {len(onset_segm)} (max. {self.n_onsets}) per buffer ({self.memo[0]/ self.memo[1]:.4%})")
            if data.max() > 0.3:
                self.line.set_color('r')
            elif (data.max() > 0.01) & (len(onset_segms) > 0):
                self.line.set_color('orange')
            else:
                self.line.set_color('g')
                        
        return self.line, self.texts
            
        
    def initialize(self):
        # xlim = self.samplerate 
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot(self.xaxis, self.plot_array, color='y')
        self.texts = self.ax.text(0.1, 0.1, "", transform=self.ax.transAxes)
        self.ax.set_ylim(-1.0, 1.0)
        # self.ax.set_xlim(0, )
        
    