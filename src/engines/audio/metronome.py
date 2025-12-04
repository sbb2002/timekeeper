import threading
import numpy as np
import sounddevice as sd

import os, sys
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".."))
sys.path.append(ROOT_PATH)

from common.handler import PrintHandler


class MetronomeWorker(PrintHandler):
    """
    """
    
    def __init__(self,
                 bpm=60,
                 subnote_per_note=4,
                 samplerate: int=44100,
                 blocksize: int=1024,
                 channels: int=1,
                 device: int | None=None):
        
        self.bpm = bpm
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.subnote_per_note = subnote_per_note
        
        # States of playing note
        self.n_frames = 0
        
        self.state_residual_first = False
        self.state_residual_onbeat = False
        self.state_residual_sub = False
        self.cutoff_first = None
        self.cutoff_onbeat = None
        self.cutoff_sub = None
        
        # Timing
        self.second_per_note = 60 / bpm
        samples_per_note = int(samplerate * self.second_per_note)

        # Number of samples / (measure, note, subnote)
        self.samples_per_measure = 4 * samples_per_note     # When BPM=60, 4 samplerate = 44100 * 4
        self.samples_per_note = samples_per_note            # When BPM=60, 1 samplerate = 44100
        self.samples_per_subnote = int(samples_per_note / subnote_per_note)     # When BPM=60, 0.25 samplerate = 44100 / 4
        # print(bpm, self.samples_per_measure, self.samples_per_note, self.samples_per_subnote )
        
        # Notes event control
        self.evt_first = threading.Event()
        self.evt_onbeat = threading.Event()
        self.evt_sub = threading.Event()
        self.evt_first.set()
        self.evt_onbeat.clear()
        self.evt_sub.clear()
        
        # Onbeat signal
        self.note_first = self.generate_sound("first")
        self.note_onbeat = self.generate_sound("onbeat")
        self.note_sub = self.generate_sound("sub")
        
        # Output thread
        self.thread = sd.OutputStream(
            samplerate=samplerate,
            blocksize=blocksize,
            device=device,
            channels=channels,
            latency='low',
            callback=self.metronome_callback
        )
        
        # Start thread
        self.t_start = self.thread.time
        self.thread.start()
        
        
    def metronome_callback(self, outdata, frames, time_info, status):
        """Metronome callback. This notify you onbeat timing."""
        
        # If status, print that.
        if status:
            self.prtwl(status)
            
        # Try to tick onbeat, or print warning.
        outdata.fill(0)
        self.ix_sample = np.linspace(
            self.n_frames, 
            self.n_frames + frames, 
            frames, dtype=np.int32)
        
        outdata_first, self.state_residual_first, self.cutoff_first = self._play_note(
            outdata, self.state_residual_first, self.cutoff_first, note="first")
        outdata_onbeat, self.state_residual_onbeat, self.cutoff_onbeat = self._play_note(
            outdata, self.state_residual_onbeat, self.cutoff_onbeat, note="onbeat")
        outdata_sub, self.state_residual_sub, self.cutoff_sub = self._play_note(
            outdata, self.state_residual_sub, self.cutoff_sub, note="sub")
        
        outdata[:] = outdata_first + outdata_onbeat + outdata_sub
        
        if outdata.max() > 0.0:
            print(outdata.min(), outdata.max(), outdata.mean())

        self.n_frames += frames
        
        
    def generate_sound(self, note: str="onbeat"):
        """Generate metronome's ticking sound.
        This provides the sine-wave array has human-sensible length(5ms)."""

        # Note info
        note_dict = {   # freq, amp, duration
            "first": (2400, 1.0),
            "onbeat": (1800, 0.7),
            "sub": (1500, 0.3)
        }
        freq, amp = note_dict.get(note, (1800, 0.7))
                
        # Sensible time
        t = self._get_ticking_time_array()

        # Ticking sound with envelop
        y_osc = np.cos(2 * np.pi * freq * t, dtype=np.float32)
        # envelop = np.exp(-t * decaying_time)
        y = amp * y_osc
        print("Y:", y.shape)

        return y.astype(np.float32).reshape((-1, 1))
    
    def _get_ticking_time_array(self):
        sensible_time = 0.005   # 5ms
        samples_per_sensible_time = int(sensible_time * self.samplerate)
        t = np.linspace(0, samples_per_sensible_time, 
                        samples_per_sensible_time,
                        dtype=np.int16)
        return t
    
    def _play_note(self, outdata, state_residual, cutoff=None, note="first"):
        
        # Assign ix_onbeat & note
        if note == "first":
            ix_onbeat = self.ix_sample % self.samples_per_measure
            note = self.note_first
        elif note == "onbeat":
            ix_onbeat = self.ix_sample % self.samples_per_note
            note = self.note_onbeat
        elif note == "sub":
            ix_onbeat = self.ix_sample % self.samples_per_subnote
            note = self.note_sub
        else:
            self.prtwl("Warning!", "Invalid note.")
            return
        
        # If cutoff previous, current note plays residual.
        if state_residual:
            state_residual = False
            self.prtwl("Residual crossover.")
            outdata[: note.shape[0] - cutoff] = note[cutoff: ]
        
        # If cycle of note, find index of note starting & end
        cycle_onbeat = (ix_onbeat == 0).any()
        if cycle_onbeat:
            ix_tick_start = np.where(ix_onbeat == 0)[0][0]
            ix_tick_end = ix_tick_start + note.shape[0]
            print("TICK: ", ix_tick_start, ix_tick_end)
            
            # If occurs cutoff current, residual lefts
            if ix_tick_end >= self.blocksize:
                state_residual = True
                cutoff = self.blocksize - ix_tick_start
                print("CUTOFF", cutoff)
                outdata[ix_tick_start: ] = note[: cutoff]
            
            # Else, just play note.
            else:
                outdata[ix_tick_start: ix_tick_end] = note

        return outdata, state_residual, cutoff
    
    
if __name__ == "__main__":
    worker = MetronomeWorker(bpm=120)
    
    while True:
        try:
            pass
        except KeyboardInterrupt:
            print("Keyboard interruption.")
            break
        
    worker.stop()
    worker.close()