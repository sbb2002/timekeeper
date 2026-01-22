import numpy as np


class PitchDetector:
    def __init__(self, sr=44100, windowsize=1024, threshold=0.15):
        """
        Yin Algorithm을 통해 pitch를 분석하는 디텍터.
        Cent값이 +/- 100 틀어지면 반음 차이나는 것임.

    
        :param sr: Sample rate. Default=`44100`.
        :param windowsize: Yin's window size. Default=`1024`.
        :param threshold: CMNDF's threshold. Default=`0.15`.
        """

        self.sr = sr
        self.windowsize = windowsize
        self.threshold = threshold

    def _flatten_audio_frame(self, audio_frame):
        return np.asarray(audio_frame, dtype=np.float64).flatten()

    def _get_tau_max(self, audio_frame):
        return len(audio_frame) - self.windowsize - 1
        
    def _get_difference_function(self, audio_frame, tau_max):

        W = self.windowsize
        diff = np.zeros(tau_max)
        for tau in range(1, tau_max):
            tmp = audio_frame[: W] - audio_frame[tau: tau + W]
            diff[tau] = np.sum(tmp**2)

        return diff
    
    def _get_cmndf(self, diff, tau_max):

        cmndf = np.ones(tau_max)
        running_sum = 0
        for tau in range(1, tau_max):
            running_sum += diff[tau]
            cmndf[tau] = diff[tau] / ((1 / tau) * running_sum)
        
        return cmndf
    
    def _find_possible_peak_candidates(self, cmndf):
        return np.where(cmndf < self.threshold)[0]

    def _find_first_peak(self, cmndf, candidates, tau_max):
        
        actual_tau = candidates[0]
        for i in range(candidates[0], tau_max - 1):
            if cmndf[i] < cmndf[i + 1]:
                actual_tau = i
                break

        return actual_tau
        
    def _interpolate_parabola(self, cmndf, actual_tau, tau_max):

        if 0 < actual_tau < tau_max - 1:
            y0, y1, y2 = cmndf[actual_tau - 1], cmndf[actual_tau], cmndf[actual_tau + 1]
            denom = 2 * y1 - (y0 + y2)
            p = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-10 else 0
            return self.sr / (actual_tau + p)
        return self.sr / actual_tau

    def _hz_to_note(self, pitch):

        if pitch <= 0:
            return None, 0

        notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 
                 'D#', 'E', 'F', 'F#', 'G', 'G#']
        
        tone = 12 * np.log2(pitch / 440.0)
        tone_rounded = round(tone)
        cents = int((tone - tone_rounded) * 100)
        note_name = notes[tone_rounded % 12]
        octave = (tone_rounded + 9) // 12 + 4

        return f"{note_name}{octave}", cents

    def detect_pitch(self, audio_frame):
        
        # 1) Get flatten audio & tau_max
        audio_frame = self._flatten_audio_frame(audio_frame)
        tau_max = self._get_tau_max(audio_frame)

        if tau_max <= 0:
            return 0    # If (-), no signal as returning 0
        
        # 2) Get the Difference Function & CMNDF
        diff = self._get_difference_function(audio_frame, tau_max)
        cmndf = self._get_cmndf(diff, tau_max)

        # 3) Peak Searching
        candidates = self._find_possible_peak_candidates(cmndf)
        if len(candidates) > 0:
            actual_tau = self._find_first_peak(cmndf, candidates, tau_max)
            pitch = self._interpolate_parabola(cmndf, actual_tau, tau_max)
            return pitch
        
        return 0    # If no candidate, no signal as returning 0
        
    def detect_scale(self, audio_frame):

        pitch = self.detect_pitch(audio_frame)
        note, cent = self._hz_to_note(pitch)

        return note, cent