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
        self.n_frames = 0       # total frames from callback
        self.second_per_note = 60 / bpm
        self.samples_per_note = int(round(samplerate * self.second_per_note))
        self.flag_remaining = False
        self.info_remaining = 0
        
        # Measure, Note, Subnote
        self.sound_note = self.generate_sound(note="onbeat")


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
           
        # Just add tick array into outdata
        outdata.fill(0)
        ## if current frame range has onbeat timing, play note
        frame_distance_at_block_start = self.n_frames % self.samples_per_note
        frame_distance_at_block_finish = (self.n_frames + frames - 1) % self.samples_per_note
        if frame_distance_at_block_start > frame_distance_at_block_finish:
            ## find onbeat timing
            n_frames_on_next_note = self.n_frames // self.samples_per_note + 1
            remaining_frames = n_frames_on_next_note * self.samples_per_note
            ix_onbeat = remaining_frames - self.n_frames
            # frame_range = np.arange(self.n_frames, self.n_frames + frames) % self.samples_per_note
            # ix_onbeat = np.argmax(frame_range == 0)
            ## remaining check
            len_sound = self.sound_note.shape[0]
            remaining = ix_onbeat + len_sound - frames
            if remaining > 0:
                ## need next block to play ticking
                self.prtwl("Index: ", ix_onbeat, "Tick len: ", len_sound, "Remaining: ", remaining)
                outdata[ix_onbeat: ] = self.sound_note[: -remaining]
                self.info_remaining = remaining
                self.flag_remaining = True
            else:
                ## can play ticking in this block at all
                outdata[ix_onbeat: ix_onbeat + len_sound] = self.sound_note

        # Remaining
        if self.flag_remaining:
            ## if remaining
            if self.info_remaining > frames:
                ## still need next block
                outdata[: ] = self.sound_note[-self.info_remaining: - (self.info_remaining - frames)]
                self.info_remaining -= frames
            else:
                outdata[: self.info_remaining] = self.sound_note[-self.info_remaining: ]
                self.flag_remaining = False
                self.info_remaining = 0

        self.play_q.put_nowait(outdata.copy())
        # print("Queued!")

        self.n_frames += frames
        
    def generate_sound(self, note: str="onbeat"):
        """Generate metronome's ticking sound.
        This provides the sine-wave array has human-sensible length(5ms)."""

        # Note info
        note_dict = {   # freq, amp, duration
            "first": (2400, 1.0),
            "onbeat": (1800, 0.4),
            "sub": (1500, 0.3)
        }
        freq, amp = note_dict.get(note, note_dict['onbeat'])
                
        # Sensible time
        t = self._get_ticking_time_array()

        # Ticking sound with envelop
        y_osc = np.cos(2 * np.pi * freq * t, dtype=np.float32)
        # envelop = np.exp(-t * decaying_time)
        y = amp * y_osc
        # print("Y:", y.shape)

        return y.astype(np.float32).reshape((-1, 1))
    
    def _get_ticking_time_array(self):
        sensible_time = 0.05   # 5ms
        samples_per_sensible_time = int(sensible_time * self.samplerate)
        t = np.linspace(0, samples_per_sensible_time - 1, 
                        samples_per_sensible_time,
                        dtype=np.int16)
        return t
    
    # def _play_note(self, outdata, residual, cutoff=None, notetype="first"):
