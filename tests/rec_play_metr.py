import sys
import librosa
import queue
import numpy as np
import sounddevice as sd
import pyqtgraph as pg
from scipy.signal import stft
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
from collections import deque


# 디버깅을 위한 위치출력 처리
class PrintHandler:
    def prtwl(self, *text):
        '''Print-With-Location'''
        print(f"[{self.__class__.__name__}] ", *text)

# 메트로놈 클래스
class Metronome:
    def __init__(self, bpm=60, samplerate=44100, blocksize=1024):
        
        # Settings
        self.bpm = bpm
        self.samplerate = samplerate # 변수명 변경: self.sample -> self.samplerate
        self.blocksize = blocksize   # 변수명 변경: self.blk -> self.blocksize
        self.f_tick = 2093           # C7
        self.decay_time = 0.05       # 50ms 감쇠 시간 (예시)
        
        # Metronome effect sound
        self.mtr_tick = self.generate_sound()
        
    def generate_sound(self):
        '''1 블록에 해당하는 메트로놈 띡 소리'''
        # 0부터 blocksize 샘플까지의 시간 배열 (초)
        t = np.linspace(0, self.blocksize / self.samplerate, self.blocksize, endpoint=False)
        
        # 주파수 성분: 코사인파
        y_osc = np.cos(2 * np.pi * self.f_tick * t)
        
        # 감쇠 성분: 짧은 길이의 지수 감쇠 (엔벨로프)
        # 소리가 50ms 내에 0으로 빠르게 줄어들도록 지수함수 적용
        envelope = np.exp(-t / self.decay_time)
        
        # 두 성분을 곱하여 '틱' 소리 생성
        y = y_osc * envelope
        
        # sounddevice는 보통 float32를 사용하므로 dtype 지정
        return y.astype(np.float32)

    def get_stamp(self):
        '''현재 BPM 기준으로 정박(4분음표) 시점을 sample index로 반환'''
        samples_per_beat = (self.samplerate * 60) / self.bpm
        return int(samples_per_beat)

# 녹음 & 재생 & 메트로놈 스레드 제어 클래스(sounddevice의 duplex stream으로 구성)
class RecordPlayMetronomeWorker(Metronome, PrintHandler):
    def __init__(self, samplerate=44100, blocksize=1024, device=None, channels=1,
                 bpm=60):
        '''녹음, 재생 스레드를 생성 및 제어하는 인스턴스를 생성함.'''
        
        super().__init__(bpm, samplerate, blocksize)
        
        # 분석용 오디오 버퍼
        self.audio_buffer = queue.Queue()
        
        # 양방향 스레드 정의
        self.thread = sd.Stream(
            samplerate=samplerate,
            blocksize=blocksize,
            device=device,
            channels=channels,
            callback=self.duplex_callback
        )
        
        # 1. 4분음표 하나의 샘플 수 계산
        # 샘플레이트 * (60초 / BPM) -> 정박(4분음표) 하나의 시간(초)을 샘플수로 변환
        samples_per_beat = (self.samplerate * 60) / self.bpm
        
        # 2. 4분음표 하나의 블록 수 계산 (Float)
        # 정박(4분음표)이 몇 개의 블록마다 돌아오는지
        self.blocks_per_beat = samples_per_beat / self.blocksize
        
        # 3. 누적 블록 카운트 (Float)
        # 블록이 지날 때마다 blocks_per_beat만큼 누적되는 값
        # 이 값이 1.0을 넘을 때 정박을 알림
        self.beat_time_accumulator = 0.0 
        
        # 4. 정박 소리 재생 플래그
        self.play_tick = False
        
        # 5. 전체 샘플 카운트
        # self.sample_counter = 0
        
    def duplex_callback(self, indata, outdata, frames, time_info, status):
        '''녹음, 재생(녹음소리+메트로놈)을 양방향으로 처리하고 녹음된 버퍼를 기록.
        추후 분석 스레드에서 저장된 버퍼를 활용함.'''
        
        # print("duplex_callback called")
        
        # Status print
        if status:
            self.prtwl("Status : ", status)

        # 1. 메트로놈 정박 시점 계산 및 플래그 업데이트
        # 매 블록마다 누적 카운트에 1.0을 더합니다.
        # 누적 값이 self.blocks_per_beat를 넘어서면 정박이 지난 것입니다.
        self.beat_time_accumulator += 1.0 
        
        if self.beat_time_accumulator >= self.blocks_per_beat:
            # 정박 시점! 누적 카운트 초기화 (다음 정박까지 남은 시간을 유지)
            # 예: blocks_per_beat = 2.5
            # 1.0, 2.0 (재생 안 함)
            # 3.0 (정박 도래) -> 3.0 - 2.5 = 0.5 (남은 시간)
            self.beat_time_accumulator -= self.blocks_per_beat
            self.play_tick = True
            
        # 2. 출력 버퍼 초기화 (녹음 소리 + 메트로놈 소리)
        # 녹음된 소리를 그대로 출력 버퍼에 복사
        outdata[:] = indata 
        
        # 3. 정박 플래그가 True일 때만 메트로놈 소리 추가
        if self.play_tick:
            # 출력 버퍼에 메트로놈 소리를 더합니다.
            # indata와 mtr_tick의 채널/차원을 고려하여 브로드캐스팅 가능하게 조정해야 합니다.
            # 현재 mtr_tick이 1차원(블록크기)이라고 가정하고, outdata가 2차원 [샘플수, 채널수] 라고 가정할 때,
            # 채널 수만큼 복사하여 더해줍니다.
            for channel in range(outdata.shape[1]):
                 outdata[:, channel] += self.mtr_tick * 0.5 # 0.5는 볼륨 조절

            self.play_tick = False # 소리 재생 후 플래그 초기화
        
        # 오디오 버퍼에 메트로놈 소리없이 녹음소리만 저장
        try:
            self.audio_buffer.put_nowait(indata.copy())
        except queue.Full:
            self.prtwl("Warning : Audio buffer is full! This block will be ignored.")
        except Exception as e:
            self.prtwl("Warning : ", e)
            
        # 샘플 카운팅
        # self.sample_counter += frames
        # if self.sample_counter >= self.samplerate:
        #     self.sample_counter -= self.samplerate
            
        #     # 정박
        #     ix_onbeat = self.sample_counter
        #     indata.copy()[ix_onbeat]
        #     self.prtwl("1 second of audio processed.")
        

# 분석 스레드(threading으로 구성)
class DynamicPlotter(QMainWindow, PrintHandler):
    def __init__(self, audio_buffer, sample_rate=44100, blocksize=1024):
        self.app = QApplication([])
        
        super().__init__()
        self.setWindowTitle("Realtime Audio Analysis")
        self.setGeometry(200, 200, 800, 600)
        
        # Settings
        self.audio_buffer = audio_buffer
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        self.win = pg.GraphicsLayoutWidget(
            show=True)
        
        # Waveplot 설정
        self.p1 = self.win.addPlot(row=0, col=0, colspan=2,
                                   title="Waveplot")
        self.plotline = self.p1.plot(pen='y')
        # self.p1.setXRange(0, x_axis[-1])
        self.p1.setYRange(-1.0, 1.0)
        self.p1.setLabel('bottom', 'Samples', units='samples')
        self.p1.setLabel('left', 'Amplitude', units='')
        
        # YIN
        self.label_f0 = QLabel("Estimated F0: N/A")
        self.label_f0.setFixedHeight(30)
        main_layout.addWidget(self.label_f0)
        # self.p2 = self.win.addPlot(row=1, col=0, colspan=2)
        # self.img = pg.ImageItem()
        # self.p2.addItem(self.img)
        # self.p2.setLabel('bottom', 'Time', units='s')
        # self.p2.setLabel('left', 'Frequency', units='Hz')
        
        # self.hist = pg.HistogramLUTItem()
        # self.win.addItem(self.hist, row=0, col=2, rowspan=2)
        # self.hist.setImageItem(self.img)
        
            
        main_layout.addWidget(self.win)
        self.show()
        
        # Timer 설정
        self.timer = QtCore.QTimer()
        self.timer.setInterval(int(1/43*1000))
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()
        
        # Audio buffer for plotting
        # maxlen = int(self.sample_rate / self.blocksize * 3)  # 5초치 버퍼
        maxlen = 32
        self.plot_buffer = deque(maxlen=maxlen)
            
    def update_plot(self):

        # 오디오 버퍼에서 가능한 모든 블록을 플롯 버퍼로 이동
        while True:
            try:
                block = self.audio_buffer.get_nowait()
                self.plot_buffer.append(block)
            except queue.Empty:
                break
            except Exception as e:
                self.prtwl("Buffer Fetch ", e)
                break
        
        # 플롯 버퍼가 비어있으면 리턴
        try:
            ori_y = np.concatenate(self.plot_buffer).reshape((-1))
            y = ori_y[::]  # 다운샘플링 (옵션)
        except:
            return
        
        # Subplot 1: Waveplot
        try:
            # wave
            x_axis = np.arange(0, y.shape[0])
            self.plotline.setData(x_axis, y)

        except Exception as e:
            self.prtwl("WAVE ", e)
        
        # # Pitch Estimation using YIN
        # f0 = librosa.yin(
        #     y, 
        #     fmin=librosa.note_to_hz('C2'), 
        #     fmax=librosa.note_to_hz('C7'), 
        #     sr=self.sample_rate, 
        #     frame_length=self.blocksize, 
        #     hop_length=512
        # )
        
        # for i, freq in enumerate(f0):
        #     midi_note = frequency_to_midi_note(freq)
            
        #     # 미디 노트가 0이 아닐 때만 출력 (음성/유효한 음높이가 감지된 경우)
        #     if midi_note > 0:
        #         note_name = librosa.midi_to_note(midi_note, cents=True)
        #         self.label_f0.setText(f"Estimated F0: {freq:.2f} Hz -> {note_name}")    
        #         # print(f"프레임 {i * 512 // self.sample_rate:.2f}s: F0={freq:.2f} Hz -> {note_name}")
        #     else:
        #         # print(f"프레임 {i * 512 // self.sample_rate:.2f}s: (No Pitch/Noise)")
        #         self.label_f0.setText(f"Estimated F0: - Hz -> --")    
                
        # Onset
        # onset_frames = librosa.onset.onset_detect(y=ori_y, units='frames')
        # # onset_samples = librosa.frames_to_samples(onset_frames)
        # print(onset_frames)
        
        # # Subplot 2: STFT
        # try:
        #     # stft
        #     f, t, zxx = stft(ori_y,
        #                      fs=self.sample_rate,
        #                      nperseg=256,
        #                      )
        #     mag = np.abs(zxx)
        #     log_mag = 20 * np.log10(mag + 1e-6)
            
        #     # self.hist.setImageItem(self.img)
        #     self.img.setImage(log_mag.T)

        # except Exception as e:
        #     self.prtwl("STFT ", e)


def frequency_to_midi_note(freq):
    """주파수를 미디 노트 번호로 변환합니다."""
    # 20 Hz 미만은 무음(Noise)으로 간주
    if freq < 20.0:
        return 0 
    # MIDI Note = 12 * log2(F0 / 440 Hz) + 69
    midi_note = 12 * np.log2(freq / 440.0) + 69
    return int(np.round(midi_note))
            
# 실행
if __name__ == "__main__":
    rpm_worker = RecordPlayMetronomeWorker()
    plot_worker = DynamicPlotter(rpm_worker.audio_buffer)
    rpm_worker.thread.start()
    sys.exit(plot_worker.app.exec_())
    
    
    print("Recording start!")
    try:
        while True:
            sd.wait()
    except KeyboardInterrupt:
        print("Stopped by user.")
    
    rpm_worker.thread.stop()
    rpm_worker.thread.close()