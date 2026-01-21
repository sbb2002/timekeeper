import threading
import queue
import time
import signal
import sys
import numpy as np

# PyQt / pyqtgraph 라이브러리 임포트
from PyQt5 import QtWidgets, QtCore 
import pqtgraph as pg 
import sounddevice as sd

# --- 오디오 및 플롯 설정 ---
SAMPLE_RATE = 44100      # 샘플링 속도 (Hz)
CHANNELS = 1             # 모노
BLOCKSIZE = 1024         # 콜백당 처리할 프레임 수
LATENCY_TARGET = 0.02    # 20ms 지연 시간 목표

PLOT_DURATION_SECONDS = 3  
MAX_POINTS = int(SAMPLE_RATE / BLOCKSIZE * PLOT_DURATION_SECONDS) * BLOCKSIZE 

# --- 공유 자원 (Queue) ---
RAW_AUDIO_QUEUE = queue.Queue()
PLAYBACK_QUEUE = queue.Queue()
WAVEFORM_QUEUE = queue.Queue()

# --- 스레드 제어 플래그 및 전역 버퍼 ---
running_flag = threading.Event()
running_flag.set()
plot_data = np.zeros(MAX_POINTS, dtype='float32')

# --- 시그널 핸들러 함수 (Ctrl+C) ---
def signal_handler(sig, frame):
    """Ctrl+C 신호 핸들러: 플래그를 해제하여 스레드 및 GUI 종료 유도"""
    print("\n\n🚨 Ctrl+C 감지! 안전한 종료를 시작합니다...")
    running_flag.clear() 

# --- sounddevice 콜백 함수 ---

def callback_in(indata, frames, time_info, status):
    """녹음 콜백: RAW_AUDIO_QUEUE에 원시 데이터를 넣습니다."""
    if status:
        print(f"Callback Status (In): {status}", file=sys.stderr)
    RAW_AUDIO_QUEUE.put(indata.copy())

def callback_out(outdata, frames, time_info, status):
    """재생 콜백: PLAYBACK_QUEUE에서 처리된 데이터를 가져옵니다."""
    if status:
        print(f"Callback Status (Out): {status}", file=sys.stderr)
        
    try:
        chunk = PLAYBACK_QUEUE.get_nowait()
        
        if isinstance(chunk, str) and chunk == "DONE":
            raise sd.CallbackStop
        
        outdata[:len(chunk)] = chunk
        if len(chunk) < len(outdata):
            outdata[len(chunk):] = 0
            
    except queue.Empty:
        outdata.fill(0) 
    except sd.CallbackStop:
        outdata.fill(0)
        raise
    
    # --- QThread를 상속받는 사용자 정의 스레드 클래스 ---
class AnalysisPlaybackThread(QtCore.QThread):
    
    def __init__(self, raw_queue, playback_queue, waveform_queue, running_flag, parent=None):
        super().__init__(parent)
        self.raw_queue = raw_queue
        self.playback_queue = playback_queue
        self.waveform_queue = waveform_queue
        self.running_flag = running_flag
        
    def run(self):
        """
        QThread의 메인 루프. QThread.start() 호출 시 실행되며 높은 우선순위를 가집니다.
        """
        print("🧠 분석/재생 QThread 시작 (우선순위 높음)")
        
        # 🌟 우선순위 설정: TimeCriticalPriority로 OS에게 이 스레드를 우선 처리하도록 요청
        self.setPriority(QtCore.QThread.TimeCriticalPriority)
        
        while self.running_flag.is_set():
            try:
                # 타임아웃을 낮춰 빠른 반응 유도
                chunk = self.raw_queue.get(timeout=0.01) 
                
                # 2. 분석 로직
                rms = np.sqrt(np.mean(chunk**2))
                analyzed_data = np.clip(chunk * 2.0, -1.0, 1.0) if rms < 0.05 else chunk.copy()
                
                # 3. PLAYBACK_QUEUE에 전송 (저지연 재생)
                self.playback_queue.put(analyzed_data)
                
                # 4. WAVEFORM_QUEUE에 전송 (GUI 갱신용)
                self.waveform_queue.put(analyzed_data[:, 0]) 
                
                self.raw_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❗ 분석/재생 QThread 오류: {e}")
                self.running_flag.clear() 
                break

        self.playback_queue.put("DONE") 
        print("✅ 분석/재생 QThread 종료")
        
        # --- GUI 업데이트 함수 (QTimer에 의해 호출) ---
def update_plot():
    """
    GUI 스레드에서 10ms마다 호출되어 그래프를 갱신하며 큐 드로핑을 적용합니다.
    """
    global plot_data
    
    if not running_flag.is_set():
        timer.stop() 
        app.quit()
        return

    # 1. 큐 드로핑 (지연 시간 최소화 로직)
    new_data = []
    MAX_QUEUE_SIZE_FOR_PLOT = 5 
    
    if WAVEFORM_QUEUE.qsize() > MAX_QUEUE_SIZE_FOR_PLOT:
        data_to_drop = WAVEFORM_QUEUE.qsize() - MAX_QUEUE_SIZE_FOR_PLOT
        print(f"⚠️ 경고: 파형 큐에 {data_to_drop}개 쌓여 지연 발생! 오래된 데이터 삭제.")
        for _ in range(data_to_drop):
            try:
                WAVEFORM_QUEUE.get_nowait()
            except queue.Empty:
                break
            
    # 남아 있는 최신 데이터만 가져와 플롯
    while not WAVEFORM_QUEUE.empty():
        try:
            new_data.append(WAVEFORM_QUEUE.get_nowait())
        except queue.Empty:
            break

    if new_data:
        new_data_array = np.concatenate(new_data)
        
        # 2. 롤링 윈도우 업데이트
        plot_data[:-len(new_data_array)] = plot_data[len(new_data_array):]
        plot_data[-len(new_data_array):] = new_data_array
        
        # 3. 그래프 업데이트
        curve.setData(plot_data)
        
# --- 메인 실행 ---
if __name__ == "__main__":
    
    # 🌟 Ctrl+C (SIGINT) 신호에 signal_handler를 연결
    signal.signal(signal.SIGINT, signal_handler) 
    
    # 1. Qt 애플리케이션 및 플롯 설정
    app = QtWidgets.QApplication(sys.argv)
    
    win = pg.PlotWidget(title=f"Real-Time Audio Waveform (TimeCriticalPriority)")
    win.show()
    win.setWindowTitle('Audio Processing (Ctrl+C to stop)')
    
    # X/Y축 범위 설정
    win.setYRange(-1.0, 1.0) 
    win.setXRange(0, MAX_POINTS / SAMPLE_RATE) 
    
    # X축 데이터 생성 (시간 단위)
    x_axis = np.arange(MAX_POINTS) / SAMPLE_RATE
    curve = win.plot(x=x_axis, y=plot_data, pen='y')
    
    # GUI 업데이트를 위한 QTimer 설정
    timer = QtCore.QTimer()
    timer.timeout.connect(update_plot) 
    timer.start(10) # 10ms 주기

    # 2. 오디오 스트림 및 분석 QThread 시작
    print("📢 녹음 스트림 시작...")
    input_stream = sd.InputStream(
        samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE, channels=CHANNELS, 
        dtype='float32', latency=LATENCY_TARGET, callback=callback_in) 

    print("🔊 재생 스트림 시작...")
    output_stream = sd.OutputStream(
        samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE, channels=CHANNELS, 
        dtype='float32', latency=LATENCY_TARGET, callback=callback_out) 

    # 🌟 QThread 인스턴스 생성 및 시작
    analysis_playback_thread = AnalysisPlaybackThread(
        RAW_AUDIO_QUEUE, PLAYBACK_QUEUE, WAVEFORM_QUEUE, running_flag)
    
    print("--- 프로그램 시작 (QThread Priority 적용) ---")
    
    try:
        input_stream.start()
        output_stream.start()
        
        # 높은 우선순위로 run() 메서드 실행
        analysis_playback_thread.start() 

        # 3. Qt 이벤트 루프 시작
        sys.exit(app.exec_())

    except Exception as e:
        print(f"\n--- 메인 프로그램 오류: {e} ---")
    
    finally:
        # 4. 모든 스트림과 스레드 정리
        print("\n--- 스트림 및 스레드 정리 ---")
        
        if analysis_playback_thread.is_alive():
            analysis_playback_thread.join(timeout=1.0)
            
        if input_stream.active:
            input_stream.stop()
        input_stream.close()
        print("✅ 녹음 스트림 종료 완료.")

        if output_stream.active:
            output_stream.stop() 
        output_stream.close()
        print("✅ 재생 스트림 종료 완료.")
        
        print("--- 프로그램 종료 ---")