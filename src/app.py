from __future__ import annotations

from flask import Flask, render_template
from flask_socketio import SocketIO
import webbrowser
import threading

from engines.audio.record import RecordWorker
from engines.audio.metronome import MetronomeWorker


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

        # Settings
        bpm = settings['bpm']
        note_denominator = settings['beatBottom']
        subnote_denominator = settings['noteValue']

        print("Timekeeper START")
        
        # Workers definition
        self.recorder = RecordWorker(
            samplerate=self.samplerate,
            blocksize=self.blocksize)
        self.metronome = MetronomeWorker(
            note_denominator=note_denominator,
            subnote_denominator=subnote_denominator,
            bpm=bpm, 
            samplerate=self.samplerate, 
            blocksize=self.blocksize
            )
        
        # Update
        if self.socketio_obj is not None:
            self._update_beat_indicator()
        
    def stop_workers(self):

        if self.current_settings['running'] == False:

            # Close & GC threads
            self.recorder.thread.stop()
            self.metronome.thread.stop()
            self.recorder.thread.close()
            self.metronome.thread.close()

            self.recorder = None
            self.metronome = None

            print("Timekeeper CLOSED")

    def _update_beat_indicator(self):

        try:
            sec_per_beat = 60 / self.current_settings['bpm']
            # sam_per_beat = self.samplerate * sec_per_beat
            # current_beat = round(self.metronome.n_frames / sam_per_beat)
            # current_beat = current_beat % self.metronome.note_denominator
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

# Application
app = Flask(__name__)
socketio = SocketIO(app)
orch = AppOrchestrator(socketio_obj=socketio)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start')
def handle_start(settings):
    orch.current_settings.update(settings)
    orch.current_settings['running'] = True
    print(f"Started with BPM={settings['bpm']}, Note={settings['noteValue']}")
    orch.start_workers(settings)

@socketio.on('stop')
def handle_stop():
    orch.current_settings['running'] = False
    orch.stop_workers()




if __name__ == '__main__':
    # threading.Thread(target=start_audio, daemon=True).start()
    webbrowser.open('http://localhost:5000')
    socketio.run(app, port=5000, debug=True)