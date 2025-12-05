import threading
import numpy as np
import sounddevice as sd
from queue import Queue

from common.handler import PrintHandler


class MetronomeWorker(PrintHandler):
    """Train your rhythmic skill! This provides quantitative measuring for onbeat-attacking.
    
    **Arguments**
    * `bpm`                 -- (int | float) Beat Per Minute. Default is `60.0`.
    * `note_per_measure`    -- (int) The number of notes per a measure. Default is `4`.
    * `subnote_per_note`    -- (int | None) The number of subnotes between notes. Default is `4`.
    * `samplerate`          -- (int) Sampling rate. Default is `44100`.
    * `blocksize`           -- (int) Samples per a buffer. Default is `1024`.
    * `channels`            -- (int) Output channels. Default is `1` as mono.
    * `device`              -- (int | None) Output device. Default is `None`.
    
    **Example**
    
    * How to start?
    
    >>> metr = MetronomeWorker()
    >>> [ANOTHER FUNCTINOAL CODE EXCEPT METRONOME]
    >>> metr.stop()
    >>> metr.close()
    
    """
    
    def __init__(self,
                 bpm: int | float=60,
                 note_per_measure: int = 4,
                 subnote_per_note: int | None=4,
                 samplerate: int=44100,
                 blocksize: int=1024,
                 channels: int=1,
                 device: int | None=None):
        
        # Arguments
        self.bpm = bpm
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.note_per_measure = note_per_measure
        self.subnote_per_note = subnote_per_note if subnote_per_note is not None else 0
        
        # Timing info
        self.n_frames = 0
        self.second_per_note = 60 / bpm
        samples_per_note = int(samplerate * self.second_per_note)
        
        # First note at measure
        self.residual_first = False
        self.cutoff_first = None
        self.samples_per_measure = note_per_measure * samples_per_note     # When BPM=60, 4 samplerate = 44100 * 4
        self.state_first = threading.Event()
        self.state_first.set()
        self.note_first = self.generate_sound("first")
        
        # Other notes at measure
        self.residual_onbeat = False
        self.cutoff_onbeat = None
        self.samples_per_note = samples_per_note            # When BPM=60, 1 samplerate = 44100
        self.state_onbeat = threading.Event()
        self.state_onbeat.clear()
        self.note_onbeat = self.generate_sound("onbeat")
        
        # Sub-notes between notes
        if subnote_per_note != 0:
            self.residual_sub = False
            self.cutoff_sub = None
            self.samples_per_subnote = int(samples_per_note / subnote_per_note)     # When BPM=60, 0.25 samplerate = 44100 / 4
            self.state_sub = threading.Event()
            self.state_sub.clear()
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
        self.thread.start()
        
        # Debugging queue
        self.play_q = Queue()
        
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
        
        # Sound plays
        outdata_first, self.residual_first, self.cutoff_first = self._play_note(
            outdata, self.residual_first, self.cutoff_first, notetype="first")
        outdata_onbeat, self.residual_onbeat, self.cutoff_onbeat = self._play_note(
            outdata, self.residual_onbeat, self.cutoff_onbeat, notetype="onbeat")
        outdata_sub, self.residual_sub, self.cutoff_sub = self._play_note(
            outdata, self.residual_sub, self.cutoff_sub, notetype="sub")
        
        # Sound mixing
        if self.state_first.is_set():
            outdata[:] = outdata_first
        elif self.state_onbeat.is_set():
            outdata[:] = outdata_onbeat
        elif self.state_sub.is_set():
            outdata[:] = outdata_sub

        # Debug info                
        if outdata.max() > 0.0:
            print(outdata.min(), outdata.max(), outdata.mean())

        self.play_q.put_nowait(outdata.copy())
        print("Queued!")

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
    
    def _play_note(self, outdata, residual, cutoff=None, notetype="first"):
        
        # Assign ix_onbeat & note
        if notetype == "first":
            ix_onbeat = self.ix_sample % self.samples_per_measure
            note = self.note_first
        elif notetype == "onbeat":
            ix_onbeat = self.ix_sample % self.samples_per_note
            note = self.note_onbeat
        elif notetype == "sub":
            ix_onbeat = self.ix_sample % self.samples_per_subnote
            note = self.note_sub
        else:
            self.prtwl("Warning!", "Invalid note.")
            return
        
        # If cutoff previous, current note plays residual.
        if residual:
            residual = False
            self.prtwl("Residual crossover.")
            outdata[: note.shape[0] - cutoff] = note[cutoff: ]
        
        # If cycle of note, find index of note starting & end
        state_onbeat = (ix_onbeat == 0).any()
        if state_onbeat:
            ix_tick_start = np.where(ix_onbeat == 0)[0][0]
            ix_tick_end = ix_tick_start + note.shape[0]
            # print("TICK: ", ix_tick_start, ix_tick_end)
            
            # If occurs cutoff current, residual lefts
            if ix_tick_end >= self.blocksize:
                residual = True
                cutoff = self.blocksize - ix_tick_start
                # print("CUTOFF", cutoff)
                outdata[ix_tick_start: ] = note[: cutoff]
            
            # Else, just play note.
            else:
                outdata[ix_tick_start: ix_tick_end] = note
                
        # Timing control
        if state_onbeat:
            if notetype == "first":
                self.state_first.set()
            elif notetype == "onbeat":
                self.state_onbeat.set()
            elif notetype == "sub":
                self.state_sub.set()
        else:
            if notetype == "first":
                self.state_first.clear()
            elif notetype == "onbeat":
                self.state_onbeat.clear()
            elif notetype == "sub":
                self.state_sub.clear()

        return outdata, residual, cutoff
