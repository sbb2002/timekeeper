import queue
import sounddevice as sd

from common.handler import PrintHandler


class RecordWorker(PrintHandler):
    
    """Worker class for recording guitar sound as input.
    
    **Arguments**
    * `samplerate` -- Samples per a second. Default is `44100`.
    * `blocksize` -- Samples per a block. Default is `1024`.
    * `channels` -- Recording channel. Default is `1` as mono.
    * `device` -- Input device. Default is `None` as auto-selection.
    
    
    **Example**
    
    * `thread`
    
    This contains the recording thread.
    You can handle thread like this.
    
    >>> worker = RecordWorker()     # Start thread automatically with assigning
    >>> [ANOTHER FUNCTINOAL CODE EXCEPT RECORDING]
    >>> worker.thread.stop()        # When ending thread
    >>> worker.thread.close()
    
    
    * `audio_buffer`
    
    Also, you can use audio buffer as `Queue`.
    If using, you should get all in the buffer. 
    
    >>> While True:
    >>>     try:
    >>>         worker.audio_buffer.get_nowait()
    >>>     except:
    >>>         ...
    
    """
    
    def __init__(self,
                 samplerate: int=44100,
                 blocksize: int=1024,
                 channels: int=1,
                 device: int | None=None):
                
        # Audio buffer for analysis
        self.audio_buffer = queue.Queue()
        
        # Input thread
        self.thread = sd.InputStream(
            samplerate=samplerate,
            blocksize=blocksize,
            device=device,
            channels=channels,
            latency='low',
            callback=self.record_callback
        )
        
        # Start thread
        self.thread.start()
        
        
    def record_callback(self, indata, frames, time_info, status):
        """Recording callback. Those arguments are neccessary for thread."""
        
        # If status, print that.
        if status:
            self.prtwl(status)
        
        # Try to acquire audio buffers, or print warning.
        try:
            self.audio_buffer.put_nowait(indata.copy())
        except queue.Full:
            self.prtwl("Warning!", "This buffer will be ignored.")
        except Exception as e:
            self.prtwl("Warning!", e)
            
