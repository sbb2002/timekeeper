import threading
import queue
import time
import sounddevice as sd
import numpy as np
import signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 오디오 설정 ---
SAMPLE_RATE = 44100  # 샘플링 속도 (Hz)
CHANNELS = 1         # 모노
BLOCKSIZE = 1024     # 콜백당 처리할 프레임 수 (CHUNK와 유사)
LATENCY = 'low'      # 지연 시간 설정

# --- 공유 자원 (Queue) ---
# sounddevice의 콜백 함수에서 오디오 데이터를 전달받는 큐
audio_in_queue = queue.Queue()
# 분석된 결과가 재생 콜백으로 전달되는 큐 (데이터 재생을 위해 numpy 배열 사용)
audio_out_queue = queue.Queue()

# --- 시그널 핸들러 함수 ---
def signal_handler(sig, frame):
    """
    Ctrl+C (SIGINT) 신호가 발생했을 때 호출되는 함수.
    running_flag를 해제하여 분석 스레드의 종료를 유도합니다.
    """
    print("\n\n🚨 Ctrl+C 감지! 안전한 종료를 시작합니다...")
    running_flag.clear()
    
# --- 스레드 제어 플래그 ---
running_flag = threading.Event()
running_flag.set() # 프로그램 시작 시 플래그를 True로 설정

# --- sounddevice 콜백 함수 ---

def callback_in(indata, frames, time_info, status):
    """sounddevice 마이크 입력 스트림의 콜백 함수"""
    if status:
        print(f"Callback Status (In): {status}", file=sys.stderr)
    
    # 받은 데이터를 audio_in_queue에 넣습니다. (녹음 스레드로 전송)
    # indata는 numpy 배열입니다.
    audio_in_queue.put(indata.copy())

def callback_out(outdata, frames, time_info, status):
    """sounddevice 스피커 출력 스트림의 콜백 함수"""
    if status:
        print(f"Callback Status (Out): {status}", file=sys.stderr)
        
    try:
        # 재생을 위해 audio_out_queue에서 데이터를 가져옵니다.
        # queue.get_nowait()을 사용하여 논블로킹으로 처리
        chunk = audio_out_queue.get_nowait()
        
        if isinstance(chunk, str) and chunk == "DONE":
            # 종료 신호를 받으면, outdata를 0으로 채우고 예외를 발생시켜 스트림을 종료합니다.
            raise sd.CallbackStop
        
        # 가져온 데이터를 outdata 버퍼에 복사
        outdata[:len(chunk)] = chunk
        
        # 남은 버퍼는 0으로 채워 오디오 끊김 방지
        if len(chunk) < len(outdata):
            outdata[len(chunk):] = 0
            
    except queue.Empty:
        # 큐가 비어 있으면 무음으로 채웁니다.
        outdata.fill(0)
    except sd.CallbackStop:
        # 종료 신호를 받았으므로, outdata를 0으로 채우고 스트림을 멈춥니다.
        outdata.fill(0)
        raise # CallbackStop 예외를 다시 던져 스트림 종료를 알립니다.

# --- 스레드 함수 정의 ---

def analysis_thread_func():
    """audio_in_queue에서 데이터를 가져와 분석하고 audio_out_queue에 결과를 넣는 스레드 함수"""
    print("🧠 분석 스레드 시작")
    
    # 5초간만 작동하도록 시뮬레이션
    start_time = time.time()
    SIMULATION_DURATION = 5 
    
    while running_flag.is_set():
        try:
            # 녹음 콜백으로부터 데이터 청크를 가져옵니다. (blocking)
            # 스트림이 실행 중이므로 타임아웃을 짧게 줍니다.
            chunk = audio_in_queue.get(timeout=0.1) 
            
            # --- 실제 분석 로직 ---
            # numpy 배열인 chunk를 사용하여 분석 수행
            
            # 1. 간단한 볼륨 레벨 계산
            rms = np.sqrt(np.mean(chunk**2))
            
            # 2. 분석 결과 (여기서는 원본 데이터를 약간 수정하여 재생)
            # 예: 볼륨이 낮으면 2배 증폭하는 시뮬레이션
            if rms < 0.05:
                # 데이터를 2배 증폭
                analyzed_data = np.clip(chunk * 2.0, -1.0, 1.0) 
                print(f"  [분석] 증폭 ({rms:.4f} -> {(np.sqrt(np.mean(analyzed_data**2))):.4f}) 처리")
            else:
                analyzed_data = chunk.copy()
                print(f"  [분석] 일반 처리 (RMS: {rms:.4f})")
            
            # 분석된 데이터를 audio_out_queue에 넣어 재생 스트림으로 전송
            audio_out_queue.put(analyzed_data)
            
            audio_in_queue.task_done()

        except queue.Empty:
            # 타임아웃 발생 시
            if time.time() - start_time > SIMULATION_DURATION:
                print("⏳ 시뮬레이션 시간 종료.")
                running_flag.clear() # 플래그를 False로 설정하여 종료를 알립니다.
            continue
        except Exception as e:
            print(f"❗ 분석 스레드 오류: {e}")
            running_flag.clear()
            break

    # 스트림 종료를 위해 재생 큐에 "DONE" 신호를 넣습니다.
    audio_out_queue.put("DONE") 
    print("✅ 분석 스레드 종료 및 재생 스트림 종료 신호 전송")

# --- 메인 실행 ---
if __name__ == "__main__":
    import sys
    
    print(f"사용할 샘플링 속도: {SAMPLE_RATE} Hz")

    # Ctrl+C handler
    signal.signal(signal.SIGINT, signal_handler)

    # 1. 스트림 정의 (sounddevice의 Stream은 자체 스레드에서 콜백을 실행)
    
    # 입력(녹음) 스트림: 마이크 데이터를 callback_in으로 보냅니다.
    print("📢 녹음 스트림 시작...")
    input_stream = sd.InputStream(
        samplerate=SAMPLE_RATE, 
        blocksize=BLOCKSIZE, 
        channels=CHANNELS, 
        dtype='float32',
        latency=LATENCY,
        callback=callback_in
    )

    # 출력(재생) 스트림: callback_out에서 데이터를 받아 스피커로 출력합니다.
    print("🔊 재생 스트림 시작...")
    output_stream = sd.OutputStream(
        samplerate=SAMPLE_RATE, 
        blocksize=BLOCKSIZE, 
        channels=CHANNELS, 
        dtype='float32',
        latency=LATENCY,
        callback=callback_out
    )

    # 2. 분석 스레드 생성 및 시작
    analysis_thread = threading.Thread(target=analysis_thread_func, name="Analyzer")
    
    print("--- 프로그램 시작 (1개 분석 스레드 + 2개 스트림 콜백 스레드) ---")
    
    try:
        # 스트림을 시작합니다. (내부적으로 콜백을 실행하는 스레드가 생성됨)
        input_stream.start()
        output_stream.start()
        
        # 분석 스레드를 시작합니다.
        analysis_thread.start()

        # 분석 스레드가 종료될 때까지 메인 스레드는 대기합니다.
        analysis_thread.join()

    except Exception as e:
        print(f"\n--- 메인 프로그램 오류: {e} ---")
    
    finally:
        # 모든 스트림과 스레드가 종료되도록 정리합니다.
        print("\n--- 스트림 및 스레드 정리 ---")
        
        if input_stream.active:
            input_stream.stop()
        input_stream.close()
        print("✅ 녹음 스트림 종료 완료.")

        if output_stream.active:
            output_stream.stop() # 'DONE' 신호에 의해 콜백이 멈췄을 가능성이 높지만, 안전하게 호출
        output_stream.close()
        print("✅ 재생 스트림 종료 완료.")
        
        print("--- 프로그램 종료 ---")