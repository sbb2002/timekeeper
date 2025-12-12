import queue
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from common.handler import PrintHandler


# Matplotlib plotter
class MatplotlibPlotter(PrintHandler):
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
                # print("Get data")
                data_from_q.append(self.data.get_nowait())
            except queue.Empty:
                # self.prtwl("Queue was empty!")
                break
        
        if data_from_q:
            data = np.concatenate(data_from_q, axis=0)
            # print("SHAPE", data.shape, self.plot_array.shape)
            
            # Energy
            # energy = np.sqrt(np.mean(data ** 2))
            
            try:
                self.plot_array[: -len(data)] = self.plot_array[len(data):]
                self.plot_array[-len(data):] = data.reshape((-1,))
            except ValueError as e:
                self.prtwl("ValueError in plot update:", str(e))
            self.line.set_ydata(self.plot_array)
            if data.max() > 0.3:
                self.line.set_color('r')
            elif data.max() > 0.01:
                self.line.set_color('orange')
            else:
                self.line.set_color('g')
                        
        return self.line,
            
        
    def initialize(self):
        # xlim = self.samplerate 
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot(self.xaxis, self.plot_array, color='y')
        self.ax.set_ylim(-1.0, 1.0)
        # self.ax.set_xlim(0, )
        
    