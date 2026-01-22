from __future__ import annotations

from flask import Flask, render_template
from flask_socketio import SocketIO
import webbrowser

from engines.orchestrator import AppOrchestrator

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
    # orch.current_settings['running'] = True
    msg = f"Started with BPM={settings['bpm']}, Note={settings['noteValue']}"
    print(msg)

    orch.start_workers(settings)

@socketio.on('stop')
def handle_stop():
    # orch.current_settings['running'] = False
    orch.stop_workers()


if __name__ == '__main__':
    # threading.Thread(target=start_audio, daemon=True).start()
    # webbrowser.open('http://localhost:5000')
    socketio.run(app, port=5000, debug=True)

    app.run(debug=True)