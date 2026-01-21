import sys
import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
import numpy as np


# app = QApplication(sys.argv)

class DynamicPlotter:
    def __init__(self):
        self.app = QApplication([])
        self.win = pg.PlotWidget(title="PyQtGraph Dynamic Plotting Example")
        self.win.show()
        self.win.setWindowTitle('pyqtgraph example')
        self.curve = self.win.plot(pen='y') # PlotDataItem 미리 생성

        self.data = np.linspace(0, 10, 100)
        self.ptr = 0
        self.timer = QtCore.QTimer()
        self.timer.setInterval(50) # 50ms 간격으로 업데이트 (약 20 FPS)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

    def update_plot(self):
        # 주기적으로 호출되는 함수에서 데이터 업데이트
        self.data = self.data + np.random.randn(100) * 0.1
        self.ptr += 1
        self.curve.setData(self.data) # setData 메소드로 데이터 갱신

    # def start(self):
    #     self.app.exec_()

if __name__ == '__main__':
    p = DynamicPlotter()
    # p.start()
    p.win.show()
    sys.exit(p.app.exec())