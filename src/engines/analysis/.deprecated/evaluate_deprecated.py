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
        self.flag_threshold = False
        self.energy_threshold = 0.0
        self.n_sample_threshold = 0
        
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
            except queue.Empty:
                self.prtwl("Warning!", "Queue was empty.")
                continue
            
            # Evaluate
            self._set_threshold(buffer)
            
            # Evaluate punctuality
            energy = np.sqrt(buffer ** 2)
            if energy.max() > self.energy_threshold:
                self.prtwl(f"#{energy.argmax()} ENERGY: ", energy.max())
                # continue
            
    def _set_threshold(self, buffer):
        
        # If threshold is 0, set threshold by initial recording
        if self.flag_threshold is False:
            energy = np.sqrt(np.mean(buffer ** 2))
            self.energy_threshold += energy
            self.n_sample_threshold += 1
            
            # Avg by 3seconds
            if self.n_sample_threshold > (self.samplerate * 3 / 1024):
                self.energy_threshold = self.energy_threshold / (self.samplerate * 3 / 1024)
                self.energy_threshold *= 10
                self.flag_threshold = True
                self.prtwl("Threshold was set. THRES.: ", self.energy_threshold)
        
        


        # Realtime displaying...
            
    def _split_buffer(self, buffer):
        ...