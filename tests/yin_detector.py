# yin_envelope_detector.py
# 사용법: python yin_envelope_detector.py input.wav
import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter, hilbert

def rms(x):
    return np.sqrt(np.mean(x*x))

def envelope_hilbert(x, win_len):
    # frame-wise RMS envelope
    env = np.sqrt(np.convolve(x*x, np.ones(win_len)/win_len, mode='same'))
    # optional smoothing already from window
    return env

def yin(frame, sr, fmin=80, fmax=1200, thresh=0.1):
    # simple YIN implementation returning period (s) or None
    # frame: 1D numpy
    N = len(frame)
    # difference function
    max_tau = int(sr / fmin)
    min_tau = int(sr / fmax)
    max_tau = min(max_tau, N//2)
    if max_tau <= min_tau:
        return None
    d = np.zeros(max_tau+1)
    for tau in range(1, max_tau+1):
        diff = frame[:-tau] - frame[tau:]
        d[tau] = np.sum(diff*diff)
    # cumulative mean normalized difference
    cmnd = np.zeros_like(d)
    cmnd[1:] = d[1:] * np.arange(1, len(d)) / (np.cumsum(d[1:]) + 1e-16)
    # find tau where cmnd < thresh
    candidates = np.where(cmnd[min_tau:max_tau+1] < thresh)[0]
    if candidates.size == 0:
        return None
    tau = candidates[0] + min_tau
    # parabolic interpolation for better precision
    if tau+1 < len(cmnd) and tau-1 >= 1:
        y0, y1, y2 = cmnd[tau-1], cmnd[tau], cmnd[tau+1]
        denom = (y0 + y2 - 2*y1)
        if abs(denom) > 1e-12:
            shift = 0.5 * (y0 - y2) / denom
            tau = tau + shift
    freq = sr / tau
    return freq

def detect(filename):
    sr, x = wavfile.read(filename)
    if x.ndim>1: x = x.mean(axis=1)  # mono
    x = x.astype(np.float32)
    x = x / (np.max(np.abs(x)) + 1e-16)

    # parameters
    frame_size = 2048
    hop = 256
    env_win = 512
    fmin = 80
    fmax = 2000
    yin_thresh = 0.12
    offset_db = -30  # peak -30dB
    min_duration_s = 0.03
    min_duration_frames = int(min_duration_s * sr / hop)

    # compute envelope
    env = envelope_hilbert(x, env_win)
    # compute frame-wise RMS and yin
    n_frames = (len(x)-frame_size)//hop + 1
    frames_env = np.zeros(n_frames)
    frames_pitch = np.zeros(n_frames)
    frames_pitch[:] = np.nan

    for i in range(n_frames):
        s = i*hop
        frame = x[s:s+frame_size] * np.hanning(frame_size)
        frames_env[i] = rms(frame)
        f = yin(frame, sr, fmin=fmin, fmax=fmax, thresh=yin_thresh)
        if f is not None:
            frames_pitch[i] = f

    # onset/offset detection
    noise_floor = np.median(frames_env[:max(1, int(0.1*sr/hop))])  # first 100ms median
    peak = frames_env.max()
    abs_offset_thr = peak * (10**(offset_db/20.0))
    events = []
    in_note = False
    note_start = 0
    last_valid_pitch_frame = -999
    for i in range(n_frames):
        current_env = frames_env[i]
        current_pitch = frames_pitch[i]
        # onset condition
        if not in_note:
            if current_env > noise_floor * 4:  # ~ +12dB margin
                # start note
                in_note = True
                note_start = i
                last_valid_pitch_frame = i if not np.isnan(current_pitch) else -999
                # record initial pitch if available
                init_pitch = current_pitch if not np.isnan(current_pitch) else None
        else:
            # update last valid pitch
            if not np.isnan(current_pitch):
                last_valid_pitch_frame = i
            # offset cond A: env below abs threshold
            condA = current_env < abs_offset_thr
            # offset cond B: pitch lost for several frames
            condB = (i - last_valid_pitch_frame) > 4  # ~4 frames of no pitch
            if condA or condB:
                duration_frames = i - note_start
                if duration_frames >= min_duration_frames:
                    # compute average pitch across note frames
                    pitch_vals = frames_pitch[note_start:i]
                    pitch_vals = pitch_vals[~np.isnan(pitch_vals)]
                    avg_pitch = pitch_vals.mean() if pitch_vals.size>0 else None
                    events.append({
                        'start_s': note_start*hop/sr,
                        'end_s': i*hop/sr,
                        'duration_s': (i-note_start)*hop/sr,
                        'pitch_hz': float(avg_pitch) if avg_pitch is not None else None
                    })
                in_note = False

    # if still in note at end
    if in_note:
        i = n_frames-1
        duration_frames = i - note_start
        if duration_frames >= min_duration_frames:
            pitch_vals = frames_pitch[note_start:i]
            pitch_vals = pitch_vals[~np.isnan(pitch_vals)]
            avg_pitch = pitch_vals.mean() if pitch_vals.size>0 else None
            events.append({
                'start_s': note_start*hop/sr,
                'end_s': i*hop/sr,
                'duration_s': (i-note_start)*hop/sr,
                'pitch_hz': float(avg_pitch) if avg_pitch is not None else None
            })

    return events

if __name__ == "__main__":
    if len(sys.argv)<2:
        print("usage: python yin_envelope_detector.py path_to.wav")
        sys.exit(1)
    ev = detect(sys.argv[1])
    print(ev)
    for e in ev:
        print(e)
