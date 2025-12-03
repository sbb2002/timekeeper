import threading
import numpy as np
import sounddevice as sd

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
        
        # self.bpm = bpm
        # self.samplerate = samplerate
        # self.subnote_per_note = subnote_per_note
        
        # Output thread
        self.thread = sd.OutputStream(
            samplerate=samplerate,
            blocksize=blocksize,
            device=device,
            channels=channels,
            latency='low',
            callback=self.metronome_callback
        )
        
        # Onbeat signal
        self.note_first = self.generate_sound("first")
        self.note_onbeat = self.generate_sound("onbeat")
        self.note_sub = self.generate_sound("sub")
        
        # Onbeat timing
        self.n_frames = 0
        self.n_note = 0
        self.n_subnote = 0
        
        # Timing
        second_per_note = 60 / bpm
        samples_per_note = int(samplerate * second_per_note)
        
        # Notes
        self.samples_per_measure = 4 * samples_per_note
        self.samples_per_note = samples_per_note
        self.samples_per_subnote = int(samples_per_note / subnote_per_note)
        
        # Notes event control
        self.evt_first = threading.Event()
        self.evt_onbeat = threading.Event()
        self.evt_sub = threading.Event()
        self.evt_first.set()
        self.evt_onbeat.clear()
        self.evt_sub.clear()
        
        # Start thread
        self.thread.start()
        
    def metronome_callback(self, outdata, frames, time_info, status):
        """Metronome callback. This notify you onbeat timing."""
        
        # If status, print that.
        if status:
            self.prtwl(status)
            
        # Try to tick onbeat, or print warning.
        ix_subnote = np.linspace(
            self.n_frames, 
            self.n_frames + frames, 
            frames, dtype=np.int16) % self.samples_per_subnote
        ix_subnote = np.where(ix_subnote == 0)[0]
        
        
        
        self.n_frames += frames
        
        
    def generate_sound(self, note: str="onbeat"):
        """Generate metronome's tick sound.
        You can generate specific sound as first onbeat note even!"""

        # Note info        
        note_dict = {
            "first": (2400, 1.0),
            "onbeat": (1800, 0.7),
            "sub": (1500, 0.3)
        }
        freq, amp = note_dict.get(note, (1800, 0.7))
        
        duration = 0.25
        decaying_time = 0.05
        
        # Time when ticking
        length = int(self.samplerate * duration)
        t = np.linspace(0, length, length)
        
        # Ticking sound with envelop
        y_osc = np.cos(2 * np.pi * freq * t)
        envelop = np.exp(-t * decaying_time)
        y = amp * y_osc * envelop
        
        return y.astype(np.float32)