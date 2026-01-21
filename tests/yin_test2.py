import sys, os
upper_path = os.path.dirname(os.path.abspath(__file__))
upper_path = os.path.dirname(upper_path)
sys.path.append(upper_path)

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from src.engines.analysis.detection.pitch import PitchDetector
from src.engines.analysis.detection.onset import OnsetDetector, OnsetPeakPicker


# Generate the sound wave
def generate_wave(duration, freq=440, sr=44100):
    
    SAMPLE_LEN = int(duration * sr)
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

# Normalize
def normalize(y):
    print(y.dtype)
    if y.dtype == np.int16:
        return y.astype(np.float32) / 32768.0
    else:
        return y / np.abs(y).max()


if __name__ == "__main__":
    
    det_pitch = PitchDetector()
    det_onset = OnsetDetector(threshold=0.001)
    picker_onset = OnsetPeakPicker(threshold=0.05)

    y_prev = generate_wave(1, freq=110) / 100
    # y_noise = np.random.rand(400) / 100

    # y = np.concatenate([y_noise, y])
    wavpath1 = r'tests\Korg-01W-Harmonics1-E5.wav'
    wavpath2 = r'tests\Kawai-K11-CleanGtr-C3.wav'
    # wavpath = r'tests\chirp.wav'
    sr, y1 = wavfile.read(wavpath1)
    sr, y2 = wavfile.read(wavpath2)
    y = np.concatenate([y1, y1, y2])

    y = normalize(y)
    if len(y.shape) >= 2:
        y = y[:, 0]
    y = np.concatenate([y_prev[:1000], y])


    W = 4096
    INITIAL = 100
    n_steps = 200
    lx, ly, lz = [], [], []
    flag_peak = False
    flag_note = None
    # last_onset_window = -999

    for n in range(n_steps):
        try:
            y_step = y[INITIAL + n * W : INITIAL + (n+1) * W]
            is_onset, flux, adt = det_onset.detect_onset(y_step)
            is_peak = picker_onset.process(flux)

            # 이 방식으로는 동일음에 대해서는 다 놓치는 문제가 있음.
            # 크로매틱을 상정하고 있기 때문에 괜찮지만 실제 연주에 사용하기엔 무리가 있음.
            # 그래서 추후 실제 연주버전에서는 리듬 체크, 음 높이 체크 버전을 나누어야 될듯.
            if is_peak:
                note, cent = det_pitch.detect_scale(y_step)
                if (flag_note != note):
                    flag_note = note
                    # last_onset_window = n
                    print(n, is_peak, note, cent)


            # if is_onset:
            #     if is_peak & (not flag_peak):
            #         flag_peak = True
            #         flag_note = note
            #         print(n, is_peak, note, cent)
            #     elif is_peak & flag_peak:
            #         pass
            #     elif (not is_peak) & (not flag_peak):
            #         pass
            #     elif (not is_peak) & flag_peak:
            #         flag_peak = False

            lx.append(INITIAL + n * W)
            ly.append(flux)
            lz.append(adt)
        except:
            break

    # y_step1 = y[2000 : 2000 + W]
    # y_step2 = y[2000 + W : 2000 + 2 * W]
    # # print(sr, y)

    # is_onset, flux = det_onset.detect_onset(y_step1)
    # is_onset, flux = det_onset.detect_onset(y_step2)

    # print(is_onset, flux)

    # note, cent = det_pitch.detect_scale(y_step2)
    # print(note, cent)

    # plot_wave(y)
    plt.figure()
    plt.plot(lx, ly)
    plt.plot(lx, lz)
    plt.plot(y, alpha=0.1)
    plt.show()