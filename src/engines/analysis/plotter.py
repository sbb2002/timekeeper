import queue
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Matplotlib plotter
class MatplotlibPlotter:
    def __init__(self, data, samplerate, blocksize, duration=5):
        
        # Arguments
        self.data = data
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.duration = duration
        
        # Plot settings
        self.xlim = int(samplerate * duration)
        self.xaxis = np.arange(self.xlim) / samplerate
        self.plot_array = np.zeros(self.xlim, dtype=np.float32)

        self.initialize()
        
        self.ani = FuncAnimation(
            self.fig,
            self.update,
            interval=10,
            blit=True,
            cache_frame_data=False
        )
        plt.show()
        
    def update(self, frame):
        
        data_from_q = []
        while True:
            try:
                print("Get data")
                data_from_q.append(self.data.get_nowait())
            except queue.Empty:
                print("Queue was empty!")
                break
        
        if data_from_q:
            data = np.concatenate(data_from_q, axis=0)
            print("SHAPE", data.shape, self.plot_array.shape)
            self.plot_array[: -len(data)] = self.plot_array[len(data):]
            self.plot_array[-len(data):] = data.reshape((-1,))
            
            self.line.set_ydata(self.plot_array)
            
        
    def initialize(self):
        
        # xlim = self.samplerate 
        
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot(self.xaxis, self.plot_array, color='y')
        # self.ax.set_xlim(0, )
        
    