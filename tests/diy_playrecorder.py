import os, sys, time
import queue
import threading
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import pyqtgraph as pg
from scipy.signal import stft
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QVBoxLayout
from collections import deque


# Settings
SAMPLE_RATE = 44100
CHANNELS = 1
BLOCK_SIZE = 1024
DURATION = 5

# Audio buffer storage
audiobuf_q = queue.Queue()
record_result = deque(maxlen=30)

# Recorder
def record_callback(indata, frames, time_info, status):
    global audiobuf_q
    # print(status)
    
    # Recording if not stopped
    try:
        # print(f"Buffer was queued. Current qsize: {audiobuf_q.qsize()}")
        audiobuf_q.put_nowait(indata.copy())
    except queue.Full:
        print("Queue was full! This buffer will be ignored.")
    except Exception as e:
        print(e)
          
def record_stream(samplerate, channels, block_size):
    thread_rec = sd.InputStream(
        samplerate=samplerate,
        channels=channels,
        blocksize=block_size,
        callback=record_callback
    )

    return thread_rec

# Play
def play_callback(outdata, frames, time_info, status):
    global audiobuf_q, record_result
    # print(status)
    
    # Playing if not stopped
    try:
        # print(f"Buffer is reading. Current qsize: {audiobuf_q.qsize()}")
        data = audiobuf_q.get()
        # print("PLAY ", data.shape)
        record_result.append(data)
        outdata[:] = data
    except queue.Empty:
        print("Queue was empty! This buffer will be ignored.")
    except Exception as e:
        print(e)
        
def play_stream(samplerate, channels, block_size):
    thread_play = sd.OutputStream(
        samplerate=samplerate,
        channels=channels,
        blocksize=block_size,
        callback=play_callback
    )

    return thread_play

# Analaysis
class DynamicPlotter:
    def __init__(self):
        self.app = QApplication([])
        self.win = pg.GraphicsLayoutWidget(
            show=True)
        
        self.p1 = self.win.addPlot(row=0, col=0, colspan=2,
                                   title="Waveplot")
        self.p2 = self.win.addPlot(row=1, col=0, colspan=2)
    
        self.ptr = 0
        self.timer = QtCore.QTimer()
        self.timer.setInterval(int(1/24*1000))
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()
            
    def update_plot(self):
        global record_result
        self.ptr += 1
        try:
            self.p1.clear()
            # wave
            # y = np.concatenate(record_result).reshape((-1))
            # i = self.ptr - 5
            # i = max(i, 0)
            # j = self.ptr
            ori_y = np.concatenate(record_result).reshape((-1))
            y = ori_y[::10]
            # print(y.shape)
            # x_max = int(BLOCK_SIZE * DURATION)
            # print(y.shape)
            x_axis = np.arange(0, y.shape[0])
            self.p1.setXRange(0, x_axis[-1])
            self.p1.setYRange(-1.0, 1.0)
            self.p1.plot(x=x_axis, y=y, pen='y')
        except Exception as e:
            print("WAVE ", e)
            
        try:
            # stft
            self.p2.clear()
            f, t, zxx = stft(ori_y,
                             fs=SAMPLE_RATE,
                             nperseg=256,
                            #  noverlap=BLOCK_SIZE//2
                             )
            mag = np.abs(zxx)
            log_mag = 20 * np.log10(mag + 1e-6)
            
            self.img = pg.ImageItem()
            self.p2.addItem(self.img)
            # self.p2.setLabel('bottom', 'Time', units='s')
            # self.p2.setLabel('left', 'Frequency', units='Hz')
            self.hist = pg.HistogramLUTItem()
            self.hist.setImageItem(self.img)
            self.img.setImage(log_mag)
            # self.p2.addWidget(self.hist)
            
            # self.p2.setXRange(0, x_max / SAMPLE_RATE)
            # self.p2.setYRange(-1.0, 1.0)
            # self.p2.plot(x=x_axis, y=stft_result)
        except Exception as e:
            print("STFT ", e)
        

if __name__ == "__main__":
    thread_rec = record_stream(SAMPLE_RATE, CHANNELS, BLOCK_SIZE)
    thread_play = play_stream(SAMPLE_RATE, CHANNELS, BLOCK_SIZE)
    qtplot = DynamicPlotter()
    thread_rec.start()
    thread_play.start()
    sys.exit(qtplot.app.exec_())
    
    print("Recording start!")
    try:
        while True:
            # time.sleep(0.1)
            sd.wait()
    except KeyboardInterrupt:
        print("Stopped by user.")
        print(f"Current qsize: {audiobuf_q.qsize()}")
        
    thread_rec.stop()
    thread_play.stop()
    
    thread_rec.close()
    thread_play.close()
    
    print("Recording was complete.")
    
    record_result = np.concatenate(record_result)
    print("Results: \n", record_result.shape, record_result.shape[0] / BLOCK_SIZE)
    
    # plt.figure()
    # plt.plot(record_result)
    # plt.show()