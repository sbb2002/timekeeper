import queue
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from common.handler import PrintHandler


# Evaluator
def get_energy(buffer):
    return np.sqrt(buffer ** 2)

def get_odf(energy):
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
    
    return odf

def set_threshold(odf, factor=1.5, offset=0.1, window_size=20):
    # Moving average
    window_size = max(min(odf.shape[0] - 2, window_size), 12)
    local_mean = np.convolve(odf, np.ones(window_size)/window_size, mode='same')
    
    # Set threshold
    threshold = local_mean * factor + offset
    
    return threshold

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
        self.total_frames = np.int64(0)
        self.last_onset_frame = np.int64(-100)

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

            # Energy
            energy = get_energy(data)
            
            # ODF
            odf = get_odf(energy)

            ## 4) Peak picking w/ shape-correction
            threshold = set_threshold(odf, factor=1.1, offset=0.005, window_size=20)
            
            try:
                peaks_over_threshold = (odf > threshold).astype(int)
                onset_frames = np.where(np.diff(peaks_over_threshold) > 0)[0] + 1
                
                if len(onset_frames) > 0:
                    self.prtwl("Onset segm.:", onset_frames, "/ Total segm.: ", odf.shape[0])
                    self.n_onsets = max(len(onset_frames), self.n_onsets)
                
                    ## Debug: Memory onsets per buffer
                    # self.memo = (self.n_onsets, data.shape[0])   # Onsets / Buffer size in active
                    # self.memories.append(self.memo)
                    
                    # Define current onset frame: total + 1st onset
                    current_onset_frame = self.total_frames + round(onset_frames[0] / odf.shape[0] * data.shape[0])
                    
                    # Measure distance from last onset
                    distance = current_onset_frame - self.last_onset_frame
                    
                    # Avoid duplicated onset within 50 centiseconds (2205 samples at 44100 Hz)
                    if distance <= 2205:
                        raise Exception(f"Distance {distance} = Curr {current_onset_frame} - Last {self.last_onset_frame}...")
                    self.last_onset_frame = current_onset_frame
                    
                
            except ValueError:
                onset_frames = []
                self.prtwl("Too much shrinken shape. It seems CPU was overloaded. Discarding onset detection for this frame.")
            
            except Exception as e:
                onset_frames = []
                self.prtwl("Same onset detection on border:", str(e))

            finally:
                self.total_frames += data.shape[0]
                
            
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
            # self.texts.set_text(f"ONSET: {len(onset_frames)} (max. {self.n_onsets}) per buffer ({self.memo[0]/ self.memo[1]:.4%})")
            if data.max() > 0.3:
                self.line.set_color('r')
            elif (data.max() > 0.01) & (len(onset_frames) > 0):
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
        
    