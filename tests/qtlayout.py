import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QLineEdit, QLabel
)
from PyQt5.QtCore import QTimer
import pyqtgraph as pg

# PyQtGraph 배경을 흰색으로 설정 (선택 사항)
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 + PyQtGraph 통합 예제")
        self.setGeometry(100, 100, 800, 600)

        # 중앙 위젯 설정
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃 (세로 방향)
        main_layout = QVBoxLayout(central_widget)

        ## 1. PyQtGraph 플롯 위젯 설정
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Voltage (V)')
        self.plot_widget.setLabel('bottom', 'Time (s)')
        
        # 초기 데이터 설정
        self.x = np.linspace(0, 10, 100)
        self.y = np.sin(self.x * 2)
        self.data_line = self.plot_widget.plot(self.x, self.y, pen='b') # 파란색 선
        
        main_layout.addWidget(self.plot_widget, 4) # 공간 비율을 4로 설정 (넓게)

        # ---
        
        ## 2. 텍스트 및 버튼 섹션 (가로 방향 레이아웃)
        control_layout = QHBoxLayout()
        
        # 텍스트 입력 위젯
        self.input_label = QLabel("데이터 입력 (진폭):")
        self.data_input = QLineEdit("1.0")
        
        # 업데이트 버튼
        self.update_button = QPushButton("플롯 & 텍스트 업데이트")
        self.update_button.clicked.connect(self.update_all)
        
        # 컨트롤 레이아웃에 위젯 추가
        control_layout.addWidget(self.input_label)
        control_layout.addWidget(self.data_input)
        control_layout.addWidget(self.update_button)

        main_layout.addLayout(control_layout)

        # ---

        ## 3. 상태 표시 텍스트 (QLabel)
        self.status_label = QLabel("✨ 초기 상태: 업데이트 대기 중...")
        main_layout.addWidget(self.status_label, 1) # 공간 비율을 1로 설정 (좁게)
        
    
    def update_all(self):
        """버튼 클릭 시 플롯 데이터와 상태 텍스트를 업데이트하는 슬롯."""
        
        try:
            # 1. 텍스트 입력값 가져오기
            amplitude_str = self.data_input.text()
            amplitude = float(amplitude_str)
            
            # 2. PyQtGraph 데이터 업데이트
            new_y = np.sin(self.x * 2) * amplitude
            self.data_line.setData(self.x, new_y)
            
            # 3. 상태 텍스트 업데이트
            status_text = f"✅ 플롯 업데이트 완료! (새 진폭: **{amplitude:.2f}**)"
            self.status_label.setText(status_text)
            
        except ValueError:
            # 입력값이 숫자가 아닐 경우 오류 처리
            self.status_label.setText("❌ 오류: 유효한 숫자를 입력해 주세요.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())