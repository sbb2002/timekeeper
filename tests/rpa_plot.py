import threading
import queue
import time
import signal
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation # FuncAnimation 임포트
import sounddevice as sd
import numpy as np

# --- 오디오 및 플롯 설정 ---
SAMPLE_RATE = 44100
CHANNELS = 1
BLOCKSIZE = 1024
LATENCY_TARGET = 0.02 # 20ms

PLOT_DURATION_SECONDS = 3  
MAX_POINTS = int(SAMPLE_RATE / BLOCKSIZE * PLOT_DURATION_SECONDS) * BLOCKSIZE

# --- 공유 자원 (Queue) ---
RAW_AUDIO_QUEUE = queue.Queue()
PLAYBACK_QUEUE = queue.Queue()

# 🌟 새로운 큐: 분석 스레드 -> FuncAnimation (파형 데이터 전송)
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
    # FuncAnimation 종료를 위해 plt.close() 호출을 시도할 수 있지만,
    # FuncAnimation이 자체적으로 종료되도록 플래그만 사용합니다.

# --- sounddevice 콜백 함수 (이전과 동일) ---
def callback_in(indata, frames, time_info, status):
    if status:
        print(f"Callback Status (In): {status}", file=sys.stderr)
    RAW_AUDIO_QUEUE.put(indata.copy())

def callback_out(outdata, frames, time_info, status):
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

# --- 분석/재생 스레드 함수 (사용자 정의 스레드) ---
def analysis_playback_thread_func():
    print("🧠 분석/재생 스레드 시작")
    
    while running_flag.is_set():
        try:
            # 1. RAW_AUDIO_QUEUE에서 원시 데이터 가져오기
            chunk = RAW_AUDIO_QUEUE.get(timeout=0.1) 
            
            # 2. 분석 로직
            rms = np.sqrt(np.mean(chunk**2))
            analyzed_data = np.clip(chunk * 2.0, -1.0, 1.0) if rms < 0.05 else chunk.copy()
            
            # 3. PLAYBACK_QUEUE에 전송 (재생 콜백 소비용)
            PLAYBACK_QUEUE.put(analyzed_data)
            
            # 🌟 4. WAVEFORM_QUEUE에 전송 (FuncAnimation 소비용)
            WAVEFORM_QUEUE.put(analyzed_data[:, 0]) # 모노 데이터만 전송
            
            # print(f"  [분석] 처리 완료, RMS: {rms:.4f}, 큐 전송")
            
            RAW_AUDIO_QUEUE.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"❗ 분석/재생 스레드 오류: {e}")
            running_flag.clear() 
            break

    PLAYBACK_QUEUE.put("DONE") 
    print("✅ 분석/재생 스레드 종료")

# --- FuncAnimation 갱신 함수 ---
def update_plot(frame):
    """
    FuncAnimation에 의해 주기적으로 호출되어 그래프를 갱신합니다.
    """
    global plot_data
    
    if not running_flag.is_set():
        # FuncAnimation을 안전하게 종료하는 방법이 복잡하므로, 
        # 메인 루프에서 예외를 발생시키도록 합니다.
        raise StopIteration 

    # 1. 새로운 데이터 가져오기
    new_data = []
    
    # 🌟 최적화: 큐에 쌓인 데이터가 너무 많으면 버려서 지연을 줄입니다.
    MAX_QUEUE_SIZE_FOR_PLOT = 5 
    if WAVEFORM_QUEUE.qsize() > MAX_QUEUE_SIZE_FOR_PLOT:
        data_to_drop = WAVEFORM_QUEUE.qsize() - MAX_QUEUE_SIZE_FOR_PLOT
        print(f"⚠️ 경고: 파형 큐에 {data_to_drop}개 쌓여 지연 발생! 오래된 데이터 삭제.")
        for _ in range(data_to_drop):
            try:
                WAVEFORM_QUEUE.get_nowait()
            except queue.Empty:
                break
            
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
        line.set_ydata(plot_data)
        
    return line, # FuncAnimation은 튜플을 반환해야 합니다.

# --- 메인 실행 ---
if __name__ == "__main__":
    
    # 🌟 Ctrl+C (SIGINT) 신호에 signal_handler를 연결합니다.
    signal.signal(signal.SIGINT, signal_handler) 
    
    # 1. Matplotlib 플롯 설정 (메인 스레드)
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title('Real-Time Audio Waveform (FuncAnimation)')
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlim(0, MAX_POINTS / SAMPLE_RATE)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    
    x_axis = np.arange(MAX_POINTS) / SAMPLE_RATE
    line, = ax.plot(x_axis, plot_data, color='y') # plot 객체를 반환합니다.
    
    # 2. 오디오 스트림 및 분석 스레드 시작
    print("📢 녹음 스트림 시작...")
    input_stream = sd.InputStream(
        samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE, channels=CHANNELS, 
        dtype='float32', latency=LATENCY_TARGET, callback=callback_in) 

    print("🔊 재생 스트림 시작...")
    output_stream = sd.OutputStream(
        samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE, channels=CHANNELS, 
        dtype='float32', latency=LATENCY_TARGET, callback=callback_out) 

    analysis_playback_thread = threading.Thread(target=analysis_playback_thread_func, name="AnalyzerPlayer")
    
    print("--- 프로그램 시작 (Ctrl+C를 눌러 종료) ---")
    
    try:
        input_stream.start()
        output_stream.start()
        analysis_playback_thread.start()

        # 🌟 3. FuncAnimation 시작
        # interval은 갱신 주기(ms)입니다. 20ms 지연 목표를 위해 10ms로 설정합니다.
        ani = FuncAnimation(fig, update_plot, interval=10, blit=True, cache_frame_data=False)
        plt.show() # FuncAnimation은 plt.show()가 호출되면 실행됩니다.

    except StopIteration:
        # FuncAnimation의 update_plot에서 종료 신호가 발생했을 때 처리
        print("\n✅ FuncAnimation 종료 요청 수신.")
    except Exception as e:
        print(f"\n--- 메인 프로그램 오류: {e} ---")
    
    finally:
        # 4. 정리
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