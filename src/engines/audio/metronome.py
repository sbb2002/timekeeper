import threading
import numpy as np
import sounddevice as sd
from queue import Queue

from common.handler import PrintHandler


# Metronome Worker
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
                 note_denominator: int=4,
                 subnote_denominator: int | None=16,
                 samplerate: int=44100,
                 blocksize: int=1024,
                 channels: int=1,
                 device: int | None=None):
        
        # Arguments
        self.bpm = bpm
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.note_denominator = note_denominator
        self.subnote_denominator = subnote_denominator
        
        # Timing info
        self.n_frames = 0       # total frames from callback

        # Notes
        self.notemakers = []
        
        self.notemaker_first = NoteSoundMaker(
            note_denominator=1,
            level=1,
            bpm=bpm,
            samplerate=samplerate,
            blocksize=blocksize
        )
        self.notemakers.append(self.notemaker_first)
        
        self.notemaker_onbeat = NoteSoundMaker(
            note_denominator=note_denominator,
            level=2,
            bpm=bpm,
            samplerate=samplerate,
            blocksize=blocksize
        )
        self.notemakers.append(self.notemaker_onbeat)
        
        if subnote_denominator is not None:
            self.notemaker_sub = NoteSoundMaker(
                note_denominator=subnote_denominator,
                level=3,
                bpm=bpm,
                samplerate=samplerate,
                blocksize=blocksize
            )
            self.notemakers.append(self.notemaker_sub)
            
        
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
        
        # First note
        for notemaker in self.notemakers[::-1]:
            notemaker.fillin_sound(outdata, frames, self.n_frames)

        # Debugging
        self.play_q.put_nowait(outdata.copy())
        # print("Queued!")
        # if outdata.max() > 0:
            # self.prtwl(f"#{self.n_frames // self.notemaker_sub.samples_per_note} subnote energy: ", outdata.max())
        
        self.n_frames += frames


# Note-sound Maker
class NoteSoundMaker(PrintHandler):
    """Metronome sound generating class.
    
    **Arguments**
    * `note_denominator`    -- (int) Denominator of notes. Default is `4`.
    * `level`               -- (int) Sound tone & amplitude level. 
                                You can select 'hi-tone & loud' as `1` ~ 'low-tone & silent' `3`. Default is `1`.
    * `bpm`                 -- (float | int) Beat Per Minutes. Default is `60`.
    * `samplerate`          -- (int) Sampling rate. Default is `44100`.
    * `blocksize`           -- (int) Samples per a buffer. Default is `1024`.
    """
    
    def __init__(self, 
                 note_denominator: int=4,
                 level: int=1,
                 bpm: float | int=60,
                 samplerate: int=44100,
                 blocksize: int=1024):
        
        # Arguments
        self.note_denominator = note_denominator
        self.level = level
        self.samplerate = samplerate
        self.bpm = bpm
        self.blocksize = blocksize
        
        # Note unit
        self.note_duration = 4 / note_denominator
        
        # Timing info
        self.second_per_note = 60 / bpm
        self.samples_per_note = int(
            round(samplerate * self.second_per_note * self.note_duration))
        self.flag_remaining = False
        self.info_remaining = 0
        
        # Note info
        self.note_dict = self._get_noteinfo()
        
        # Measure, Note, Subnote
        self.sound_note = self.generate_sound(level=level)
        
    def fillin_sound(self, outdata, frames, n_frames):
        
        ## if current frame range has onbeat timing, play note
        frame_distance_at_block_start = n_frames % self.samples_per_note
        frame_distance_at_block_finish = (n_frames + frames - 1) % self.samples_per_note
        if frame_distance_at_block_start > frame_distance_at_block_finish:
            
            ## find onbeat timing
            n_frames_on_next_note = n_frames // self.samples_per_note + 1
            remaining_frames = n_frames_on_next_note * self.samples_per_note
            ix_onbeat = remaining_frames - n_frames

            ## remaining check
            len_sound = self.sound_note.shape[0]
            remaining = ix_onbeat + len_sound - frames
            
            if remaining > 0:
                ## need next block to play ticking
                # self.prtwl("Index: ", ix_onbeat, "Tick len: ", len_sound, "Remaining: ", remaining)
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
                outdata[:] = self.sound_note[-self.info_remaining: - (self.info_remaining - frames)]
                self.info_remaining -= frames
            else:
                outdata[: self.info_remaining] = self.sound_note[-self.info_remaining: ]
                self.flag_remaining = False
                self.info_remaining = 0
   
    def generate_sound(self, level: int=1):
        """Generate metronome's ticking sound.
        This provides the sine-wave array has human-sensible length(5ms)."""

        freq, amp = self.note_dict.get(level, self.note_dict[1])
                
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
   
    def _get_noteinfo(self):
        return {   # freq, amp, duration
            1: (2400, 0.4),     # Big sound, high tone as Lv 1
            2: (1800, 0.2),     # Medium sound, middle tone as Lv 2
            3: (1500, 0.1)      # Small sound, low tone as Lv 3
        }