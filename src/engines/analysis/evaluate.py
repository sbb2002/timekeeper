import librosa
import queue
import threading
import numpy as np

from common.handler import PrintHandler


class EvaluatorWorker(PrintHandler):
    def __init__(self, 
                 data,
                 note_denominator,
                 subnote_denominator, 
                 bpm=60, 
                 samplerate=44100
                 ):
        
        # Arguments
        self.data = data
        self.note_denominator = note_denominator
        self.subnote_denominator = subnote_denominator
        self.bpm = bpm
        self.samplerate = samplerate
        
        # Threshold settings
        self.n_frames = 0
        
        # Open the thread
        self.thread = threading.Thread(
            target=self.process_evaluation,
            daemon=True
        )

        # Start thread
        self.thread.start()

    def process_evaluation(self):
        
        while True:
            
            # Get audio buffer from recorder
            try:
                # self.prtwl("Get buffer.")
                buffer = self.data.get(timeout=0.1)
                self.n_frames += len(buffer)
                
            except queue.Empty:
                self.prtwl("Warning!", "Queue was empty.")
                continue
            
            # Get attack point
            
            ## Energy
            energy = np.sqrt(buffer ** 2)
            # self.prtwl(f"Frame #{self.n_frames} ENERGY: ", energy.shape)
            
            ## Smoothing(opt.)
            # smoothed_energy = self._smoothing(energy, windowsize=10)
            # self.prtwl("Smoothed ENERGY: ", smoothed_energy.shape)
            
            ## Onset detection
            onset = np.diff(energy, axis=0)
            if onset.max() > 0.03:
                self.prtwl("ONSET: ", onset.max(), "at frame", onset.argmax())
            
            
            
            
    def _smoothing(self, buffer, windowsize):
        window = np.ones(windowsize) / windowsize
        conv = np.convolve(buffer.reshape((-1)), v=window, mode="valid")
        return conv.reshape((-1, 1))