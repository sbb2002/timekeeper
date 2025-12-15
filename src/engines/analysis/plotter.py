import queue
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from common.handler import PrintHandler


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
        self.n_onsets = 0
        self.memories = []
        self.memo = (0, 1)

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
            data = np.concatenate(buffer, axis=0)
            # print("SHAPE", data.shape, self.plot_array.shape)
            
            # Energy
            energy = np.sqrt(data ** 2)
            # self.prtwl("ENERGY shape:", energy.shape)
            
            # Onset
            # data = np.diff(data, axis=0)
            
            ## 1) STFT
            f, t, zxx = scipy.signal.stft(energy.reshape((-1,)))
            # self.prtwl("STFT shape:", f.shape, t.shape, zxx.shape)
            
            ## 2) Magnitude
            mag = np.abs(zxx)
            # self.prtwl("STFT MAG:", mag.shape)
            
            ## 3) ODF
            prev_mag = mag[:, :-1]
            curr_mag = mag[:, 1:]
            diff_mag = curr_mag - prev_mag
            halfwave_rectified = np.maximum(0, diff_mag)
            odf = np.sum(halfwave_rectified, axis=0)
            # self.prtwl("ODF:", odf.shape)
            
            
            ## 4) Peak picking w/ shape-correction
            window_size = max(min(odf.shape[0] - 2, 20), 12)
            if window_size < 20:
                self.prtwl("Window size:", window_size)
            local_mean = np.convolve(odf, np.ones(window_size)/window_size, mode='same')
            # self.prtwl("Local mean:", local_mean.shape)
            
            threshold_factor = 1.5
            threshold_offset = 0.01
            threshold = local_mean * threshold_factor + threshold_offset
            
            try:
                peaks_over_threshold = (odf > threshold).astype(int)
                onset_frames = np.where(np.diff(peaks_over_threshold) > 0)[0] + 1
            except ValueError:
                self.prtwl("Too much shrinken shape. It seems CPU was overloaded. Discarding onset detection for this frame.")
            
            if len(onset_frames) > 0:
                self.prtwl("Onset frames:", onset_frames)
                self.n_onsets = max(len(onset_frames), self.n_onsets)
            
                ## Debug: Memory onsets per buffer
                self.memo = (self.n_onsets, data.shape[0])   # Onsets / Buffer size in active
                self.memories.append(self.memo)
            
            # if (data.max() > 0.01) & (data.max() < 0.03):
                # data = data * 10
            
            # Update plot array
            try:
                self.plot_array[: -len(data)] = self.plot_array[len(data):]
                self.plot_array[-len(data):] = data.reshape((-1,))
                
                # self.dots_array[: -len(data)] = self.dots_array[len(data):]
                # self.dots_array[-len(data):] = 
            except ValueError as e:
                self.prtwl("ValueError in plot update:", str(e))
                
            # Update line
            self.line.set_ydata(self.plot_array)
            self.texts.set_text(f"ONSET: {len(onset_frames)} (max. {self.n_onsets}) per buffer ({self.memo[0]/ self.memo[1]:.4%})")
            if data.max() > 0.3:
                self.line.set_color('r')
            elif data.max() > 0.01:
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
        
    