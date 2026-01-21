import numpy as np
import sounddevice as sd
import threading
import queue
import time
import math

# -----------------------------
# utility: note name from freq
# -----------------------------
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

def freq_to_note(freq):
    if freq is None or freq <= 0:
        return None
    midi = 69 + 12 * math.log2(freq / 440.0)
    midi_round = int(round(midi))
    name = NOTE_NAMES[midi_round % 12]
    octave = (midi_round // 12) - 1
    cents = int(round((midi - midi_round) * 100))
    return f"{name}{octave}", cents

# -----------------------------
# quadratic interpolation (parabolic) for peak refinement
# given mags at k-1, k, k+1
# -----------------------------
def parabolic_interpolation(m_minus, m0, m_plus):
    denom = (m_minus - 2*m0 + m_plus)
    if denom == 0:
        return 0.0
    delta = 0.5 * (m_minus - m_plus) / denom
    return delta  # offset in bins relative to center

# -----------------------------
# HPS-based pitch estimator (spectrum domain)
# -----------------------------
def hps_pitch_estimate(frame, sr, N=4096, H=4, fmin=70, fmax=1000):
    # window & FFT
    if len(frame) < N:
        frame = np.pad(frame, (0, N - len(frame)))
    w = np.hanning(N)
    X = np.fft.rfft(frame * w, n=N)
    mag = np.abs(X)

    # limit bins to search range
    bin_min = max(1, int(fmin * N / sr))
    bin_max = min(len(mag)-1, int(fmax * N / sr))

    # build HPS
    hps = mag.copy()
    for h in range(2, H+1):
        # downsample mag by factor h
        down = mag[::h]
        hps[:len(down)] *= down

    # search peak in restricted range
    search = hps[bin_min:bin_max+1]
    if len(search) == 0:
        return None, None  # out of range

    peak_idx = np.argmax(search) + bin_min

    # sanity check magnitude
    peak_mag = hps[peak_idx]
    if peak_mag <= 1e-8:
        return None, None

    # refine peak with 3-point parabolic interpolation on original mag
    k = peak_idx
    if 1 <= k < len(mag)-1:
        delta = parabolic_interpolation(mag[k-1], mag[k], mag[k+1])
    else:
        delta = 0.0

    precise_bin = k + delta
    freq = precise_bin * sr / N
    return freq, peak_mag

# -----------------------------
# time-domain autocorrelation fallback (fast)
# -----------------------------
def autocorr_f0(frame, sr, fmin=50, fmax=1000):
    # center clip helps sometimes - but keep simple
    x = frame - np.mean(frame)
    N = len(x)
    # compute autocorrelation via FFT (fast)
    fft = np.fft.rfft(x, n=2*N)
    acf = np.fft.irfft(np.abs(fft)**2)[:N]
    acf /= np.max(np.abs(acf)) + 1e-9

    # lag range
    lag_min = int(sr / fmax)
    lag_max = int(sr / fmin)
    if lag_max >= len(acf):
        lag_max = len(acf) - 1
    if lag_min >= lag_max:
        return None

    # find peak in range
    peak = np.argmax(acf[lag_min:lag_max+1]) + lag_min
    if acf[peak] < 0.1:
        return None
    freq = sr / peak
    return freq

# -----------------------------
# processing thread: reads frames from queue and computes pitch
# -----------------------------
def processor_thread(q, stop_event, sr=44100, N=4096, hop=2048,
                     H=4, fmin=70, fmax=1000, verbose=True):
    while not stop_event.is_set():
        try:
            frame = q.get(timeout=0.1)
        except queue.Empty:
            continue
        # convert to mono if needed
        if frame.ndim > 1:
            frame = np.mean(frame, axis=1)
        # HPS estimate
        freq, peak = hps_pitch_estimate(frame, sr, N=N, H=H, fmin=fmin, fmax=fmax)
        method = "HPS"
        # fallback to autocorr if HPS failed or freq out of range
        if freq is None or not (fmin*0.9 <= freq <= fmax*1.1):
            freq_ac = autocorr_f0(frame, sr, fmin=fmin, fmax=fmax)
            if freq_ac is not None:
                freq = freq_ac
                method = "ACF"
        if freq is None:
            if verbose:
                print("No pitch")
        else:
            note, cents = freq_to_note(freq)
            if verbose:
                print(f"[{method}] {freq:.2f} Hz — {note} ({cents} cents)")
        q.task_done()

# -----------------------------
# main: audio stream + queue + thread
# -----------------------------
def main():
    sr = 44100
    device = None  # None -> default; or put your device ID
    N = 4096
    hop = 2048
    H = 4
    fmin = 70
    fmax = 1000

    q = queue.Queue(maxsize=8)
    stop_event = threading.Event()
    t = threading.Thread(target=processor_thread,
                         args=(q, stop_event),
                         kwargs={'sr': sr, 'N': N, 'hop': hop, 'H': H, 'fmin': fmin, 'fmax': fmax},
                         daemon=True)
    t.start()

    def callback(indata, frames, time_info, status):
        if status:
            print("Stream status:", status)
        # push into queue (non-blocking; drop if queue full to keep real-time)
        try:
            q.put_nowait(indata.copy())
        except queue.Full:
            pass

    with sd.InputStream(device=device, channels=1, samplerate=sr,
                        blocksize=hop, callback=callback):
        print("Listening... Ctrl+C to stop")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping...")
    stop_event.set()
    t.join()

if __name__ == "__main__":
    main()
