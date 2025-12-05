from engines.audio.metronome import MetronomeWorker
from engines.analysis.plotter import MatplotlibPlotter

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024

if __name__ == "__main__":
    
    worker = MetronomeWorker(
        bpm=120, 
        samplerate=SAMPLE_RATE, 
        blocksize=BLOCK_SIZE)
    plotter = MatplotlibPlotter(
        data=worker.play_q,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE)
    
    while True:
        try:
            pass
        except KeyboardInterrupt:
            print("Keyboard interruption.")
            break
    
    worker.stop()
    worker.close()
    
    