from __future__ import annotations

import threading
# from engines.audio.record import RecordWorker
# from engines.audio.metronome import MetronomeWorker
from engines.audio import RecordWorker, MetronomeWorker
from engines.analysis import RhythmSupervisor


class AppOrchestrator:
    def __init__(self, socketio_obj=None):
        
        self.socketio_obj = socketio_obj
        self.samplerate = 44100
        self.blocksize = 1024

        # Settings
        self.current_settings = {
            'bpm': 120,
            'beatTop': 4,
            'beatBottom': 4,
            'noteValue': 8,
            'running': False
        }

        # Memory
        self.current_beat = 0

    def start_workers(self, settings):

        # Fool-proof: prevent to start double
        if self.current_settings['running']:
            print("Already starting.")
            return

        # Settings
        bpm = settings['bpm']
        note_denominator = settings['beatBottom']
        subnote_denominator = settings['noteValue']
        self.current_settings['running'] = True

        print("Timekeeper START")
        
        # Workers definition
        self.recorder = RecordWorker(
            samplerate=self.samplerate,
            blocksize=self.blocksize
            )
        self.metronome = MetronomeWorker(
            note_denominator=note_denominator,
            subnote_denominator=subnote_denominator,
            bpm=bpm, 
            samplerate=self.samplerate, 
            blocksize=self.blocksize
            )
        self.supervisor = RhythmSupervisor(
            target_buffer=self.recorder.buffer,
            bpm=bpm,
            subnote_denominator=subnote_denominator,
            samplerate=self.samplerate,
            flux_ratio_threshold=0.2,
            peak_threshold=0.7
        )

        self.supervisor.process_rhythm(self.socketio_obj)

        # Update
        if self.socketio_obj is not None:
            self._update_beat_indicator()
        
    def stop_workers(self):

        if self.current_settings['running']:

            self.current_settings['running'] = False

            # Close & GC threads
            self.recorder.thread.stop()
            self.metronome.thread.stop()
            self.recorder.thread.close()
            self.metronome.thread.close()

            self.kill_process()

            self.recorder = None
            self.metronome = None
            self.supervisor = None

            print("Timekeeper CLOSED")
        
        else:
            print("Already stopping.")

    def _update_beat_indicator(self):

        try:
            sec_per_beat = 60 / self.current_settings['bpm']
            current_beat = self.current_beat % self.metronome.note_denominator

            self.socketio_obj.emit(
                'beat_update', {
                    'currentBeat': current_beat
                }
            )
            self.current_beat += 1

            # print(sec_per_beat, current_beat, self.metronome.n_frames, self.metronome.n_frames / sam_per_beat)
            threading.Timer(sec_per_beat, self._update_beat_indicator).start()

        except:
            print("Worker is gone.")
            self.current_beat = 0

    def kill_process(self):
        self.supervisor.kill_process()
        self.socketio_obj.emit(
            'beat_update', {'currentBeat': -1}
        )