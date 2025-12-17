import queue
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import stft
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from common.handler import PrintHandler


# Evaluator
def get_energy(buffer):
    return np.sqrt(buffer ** 2)

def get_odf(energy):
    ## 1) STFT
    _, _, zxx = stft(energy.reshape((-1,)))
    # self.prtwl("STFT shape:", f.shape, t.shape, zxx.shape)
    
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

def merge_onsets_by_strength(onset_frames, onset_strengths, sr=44100, temporal_resolution=20):
    
    # Set frame distance
    FRAME_THRESHOLD = int(sr * temporal_resolution / 1000)
    
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
            data = np.concatenate(buffer, axis=0)
            energy = get_energy(data)
            odf = get_odf(energy)
            threshold = set_threshold(odf, factor=1.6, offset=0.1, window_size=20)
            
            try:
                onset_segms, onset_strengths = detect_onset(odf, threshold)
                onset_frames = convert_segms_into_frames(
                    onset_segms,
                    total_segms=odf.shape[0],
                    blocksize=self.blocksize
                    )
                final_onsets = merge_onsets_by_strength(
                    onset_frames, onset_strengths,
                    sr=self.samplerate,
                )
                
                if len(final_onsets) > 0:
                    self.prtwl(final_onsets)
                    if self.last_onsets.size > 0:
                        prev_onset = self.last_onsets[-1]
                        curr_onset = final_onsets[0] + self.total_frames
                        distance = curr_onset - prev_onset
                        if distance < 882:
                            self.prtwl("Distance: ", distance, f"=Curr {curr_onset} - Last {prev_onset}")
                    self.last_onsets = final_onsets + self.total_frames
                
                # peaks_over_threshold = (odf > threshold).astype(int)
                # onset_segm = np.where(np.diff(peaks_over_threshold) > 0)[0]
                
                # if len(onset_segm) > 0:
                #     # self.prtwl("Onset segm.:", onset_segm, "/ Total segm.: ", odf.shape[0])
                
                #     # All onsets are same from 1st onset to +20ms(~882frames)
                #     onset_frame = np.round(onset_segm / odf.shape[0] * data.shape[0]).astype(int)
                #     self.prtwl("Onset frame:", onset_frame, "/ Total frame: ", odf.shape[0] * data.shape[0])

                
                
                #     # Convert into current onset frame: total + 1st onset
                #     onset_frame = round(onset_segm[0] / odf.shape[0] * data.shape[0])
                #     current_onset_frame = self.total_frames + onset_frame
                    
                #     # Measure distance from last onset
                #     distance = current_onset_frame - self.last_onset_frame
                    
                #     # Avoid duplicated onset within 50 centiseconds (2205 samples at 44100 Hz)
                #     if distance <= 2205:
                #         raise Exception(f"Distance {distance} = Curr {current_onset_frame} - Last {self.last_onset_frame}...")
                #     self.last_onset_frame = current_onset_frame
                    
                
            except ValueError:
                onset_segm = []
                self.prtwl("Too much shrinken shape. It seems CPU was overloaded. Discarding onset detection for this frame.")
            
            except Exception as e:
                # onset_segm = []
                self.prtwl("Same onset detection on border: \n\t", str(e), onset_segms)

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
        
    