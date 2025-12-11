from __future__ import annotations

from engines.audio.record import RecordWorker
from engines.audio.metronome import MetronomeWorker
from engines.analysis.plotter import MatplotlibPlotter
from engines.analysis.evaluate import EvaluatorWorker

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
BPM = 60
NOTE_DENOMINATPR = 4
SUBNOTE_DENOMINATOR = 16

if __name__ == "__main__":
    
    recorder = RecordWorker(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE)
    metronome = MetronomeWorker(
        note_denominator=NOTE_DENOMINATPR,
        subnote_denominator=SUBNOTE_DENOMINATOR,
        bpm=BPM, 
        samplerate=SAMPLE_RATE, 
        blocksize=BLOCK_SIZE)
    plotter = MatplotlibPlotter(
        data=recorder.buffer,
        # data=metronome.buffer,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        duration=10)
    # evaluator = EvaluatorWorker(
    #     data=recorder.buffer,
    #     samplerate=SAMPLE_RATE,
    # )
    
    while True:
        try:
            pass
        except KeyboardInterrupt:
            print("Keyboard interruption.")
            break
    
    recorder.stop()
    metronome.stop()
    recorder.close()
    metronome.close()
    # evaluator.thread.join()
    
    