from __future__ import annotations

import queue
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import stft, decimate

from common.handler import PrintHandler
from engines.analysis.functions import *
from engines.analysis.detection.onset import OnsetDetector, OnsetPeakPicker
from engines.analysis.detection.pitch import PitchDetector


# Matplotlib plotter
class MatplotlibPlotter(PrintHandler):
    def __init__(self, data, bpm, denominator, samplerate, blocksize, duration=5):
        
        # Arguments
        self.data = data
        self.bpm = bpm
        self.denominator = denominator
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.duration = duration
        
        # Plot settings
        self.xlim = int(samplerate * duration)
        self.xaxis = np.arange(self.xlim) / samplerate
        self.plot_array = np.zeros(self.xlim, dtype=np.float32)
        # self.dots_array = np.zeros(self.xlim, dtype=np.float32)
        
        # Dectectors
        self.onset_detector = OnsetDetector(
            sr=self.samplerate,
            hop_length=128,
            threshold=0.2
            )
        self.pitch_detector = PitchDetector(
            sr=self.samplerate,
            windowsize=1024,
            threshold=0.15
        )
        self.onset_picker = OnsetPeakPicker(
            threshold=0.05,
            lookahead=1,
            lookback=1
        )

        # Memory
        self.flag_note = None
        
        # Initialize plot
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
                samples = self.data.get_nowait()
                buffer.append(samples)
                
                n_samples = sum([len(sam) for sam in samples])
                if n_samples >= 4096:
                    break
                
            except queue.Empty:
                # self.prtwl("Queue was empty!")
                break
        
        if buffer:
            # Collect all data in buffer
            data = np.concatenate(buffer, axis=0).flatten()     # [frames,]

            # Onset & Pitch detection
            is_onset, flux, adt = self.onset_detector.detect_onset(data)
            is_peak = self.onset_picker.process(flux)

            flux_arr = np.array([flux] * len(data))

            if is_peak:
                note, cent = self.pitch_detector.detect_scale(data)
                if self.flag_note != note:
                    print(f"Onset: {is_onset} ({flux:.5f})", " // ", f"Pitch: {note} {cent}")


            

            # Update plot array
            try:
                self.plot_array[: -len(data)] = self.plot_array[len(data):]
                self.plot_array[-len(data):] = flux_arr.reshape((-1,))
                
            except ValueError as e:
                self.prtwl("ValueError in plot update:", str(e))
                
            # Update line
            self.line.set_ydata(self.plot_array)
            # self.texts.set_text(f"ONSET: {len(onset_segm)} (max. {self.n_onsets}) per buffer ({self.memo[0]/ self.memo[1]:.4%})")
            if data.max() > 0.3:
                self.line.set_color('r')
            elif (data.max() > 0.01):
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
        
    def _onset_detection(self, odf, threshold):
        # Onset Detection
        onset_segms, onset_strengths = detect_onset(odf, threshold)     # [seg_ix], [seg_val]
        onset_frames = convert_segms_into_frames(
            onset_segms,
            total_segms=odf.shape[0],
            blocksize=self.blocksize
            )   # [seg]
        
        # Onset filtering in this block
        final_onsets = merge_onsets_by_strength(
            onset_frames, onset_strengths,
            sr=self.samplerate,
        )       # [seg]
        
        # Onset filtering between last and current block
        current_onset_frames, last_onsets = validate_first_onset_connecting_last_onset(
            current_onset_frames=final_onsets,
            last_onset_frames=self.last_onsets,
            total_frames=self.blocksize,
            sr=self.samplerate
        )       # [seg]

        return current_onset_frames, last_onsets

    def _pitch_detection(self):
        # If onset, Pitch Detection
        DOWNSAMPLING_FACTOR = 2
        downsampled_data = decimate(data, DOWNSAMPLING_FACTOR)
        downsampled_sr = self.samplerate / DOWNSAMPLING_FACTOR
        pitch = ultra_fast_yin(
            downsampled_data, downsampled_sr,
            threshold=0.8)
        if pitch > 0:
            # Pitch to Note
            note, cent = hz_to_note(pitch)
            
        return note, cent