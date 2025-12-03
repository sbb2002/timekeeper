import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore

app = QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="Subplots with varying sizes")

# Add a plot spanning 2 rows
p1 = win.addPlot(row=0, col=0, rowspan=2, title="Large Plot")
p1.plot([1, 2, 3], [1, 2, 1])

# Add a normal-sized plot
p2 = win.addPlot(row=0, col=1, title="Normal Plot")
p2.plot([1, 2, 3], [3, 1, 2])

# Add another normal-sized plot
p3 = win.addPlot(row=2, col=0, title="Another Normal Plot")
p3.plot([1, 2, 3], [2, 3, 1])

# Add a plot spanning 2 columns
p4 = win.addPlot(row=1, col=1, colspan=2, title="Wide Plot")
p4.plot([1, 2, 3], [1, 3, 2])


if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        sys.exit(app.exec())