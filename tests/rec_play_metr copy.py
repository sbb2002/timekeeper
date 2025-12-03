import queue
import threading
import numpy as np
import sounddevice as sd



# 디버깅을 위한 위치출력 처리
class PrintHandler:
    def prtwl(self, text):
        '''Print-With-Location'''
        print(f"[{self.__class__.__name__}] ", text)

# 메트로놈 클래스
class Metronome:
    def __init__(self, bpm=60, samplerate=44100, blocksize=1024):
        
        # Settings
        self.bpm = bpm
        self.sample = samplerate
        self.blk = blocksize
        self.f_tick = 2093  # C7
        self.decay = 0.8    # Decay rate
        
        # Metronome effect sound
        self.mtr_tick = self.generate_sound()
        
    def generate_sound(self):
        '''1블록에 해당하는 메트로놈 띡 소리'''
        t = np.linspace(0, self.blk)
        y = np.cos(2 * np.pi * self.f_tick * t) * np.exp(-1 * self.decay)
        return y



# 녹음 & 재생 & 메트로놈 스레드 제어 클래스(sounddevice의 duplex stream으로 구성)
class RecordPlayMetronomeWorker(Metronome, PrintHandler):
    def __init__(self, samplerate=44100, blocksize=1024, device=None, channels=1,
                 bpm=60):
        '''녹음, 재생 스레드를 생성 및 제어하는 인스턴스를 생성함.'''
        
        super().__int__(bpm, samplerate, blocksize)
        
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
        
        # 메트로놈 관련 정보
        self.total_frames = 0
        self.sr = samplerate
        self.blk = blocksize
        self.tick_timing = True
        self.tick_endpoint = int(1 / (self.bpm / 60) * self.sr)
        self.tick_duration = int(0.3 * self.sr)
        self.tick_frames = 0
        
        # Tick event 관리; 4가지 경우 중 실제 사용은 3가지(FF, TF, FT)
        # FF: 메트로놈이 울릴 구간이 아닌 경우
        # TF: 메트로놈이 울리기 시작할 경우
        # FT: 메트로놈이 울리기 끝날 경우
        # ** block이 지나치게 길거나 duration이 짧은 경우 TT가 나올 수 있으나 사람이 인식하기 어려움.
        self.tick_start = threading.Event()
        self.tick_end = threading.Event()
        self.tick_start.set()
        self.tick_end.clear()
        
    def duplex_callback(self, indata, outdata, frames, time_info, status):
        '''녹음, 재생(녹음소리+메트로놈)을 양방향으로 처리하고 녹음된 버퍼를 기록.
        추후 분석 스레드에서 저장된 버퍼를 활용함.'''
        # Status print
        if status:
            self.prtwl("Status : ", status)

        # 녹음 후 바로 메트로놈과 함께 재생
        
        # Tick timing 제어
        # 만약 tick_start가 T라면, tick_end가 나왔는지 안나왔는지 판단해야함
        if self.tick_start.is_set():
            # 블록 시작점, 끝점
            blk_start = max(0, self.total_frames - self.blk)
            blk_end = self.total_frames - 1
            # 나머지를 이용해 tick endpoint 도출
            mod_array = np.linspace(blk_start, blk_end, self.blk, dtype=np.int16)
            mod_shifted = np.roll(mod_array, 1)
            mod_different = mod_shifted - mod_array
            mod_different = mod_different[1:-1]
            ix_endpoint = np.where(mod_different != 1)[0]
            if len(ix_endpoint) > 0:
                # tick_end 도달
                self.tick_end.set()
                self.tick_start.clear()
                
        # 만약 tick_end가 F라면, tick_start가 나왔는지 안나왔는지 판단해야함
        
        # TF case;
        if self.tick_start.is_set() & (not self.tick_end.is_set()):
            
            ...
        # FT case
        elif (not self.tick_start.is_set()) & (not self.tick_end.is_set()):
            ...
        # FF case
        elif (not self.tick_start.is_set()) & self.tick_end.is_set():
            ...
        
        # if self.tick_timing == True:
        #     # 만약 정박이면 ticking 시작
        #     self.tick_frames += frames
        #     if self.tick_frames > self.tick_duration:
        #         # 만약 이번 frame에 ticking endpoint가 존재하면 거기까지만 넣기
        #         ix_endpoint = self.tick_frames // self.blk
        #         self.tick_timing = False
        #     else:
        #         # 아니라면 이번 블록에 통째로 tick 추가
        #         outdata[:] = indata.copy() + self.mtr_tick
        # # 메트로놈 띡 소리 기간 외인 경우 그냥 녹음소리만 재생
        # else:
        #     outdata[:] = indata.copy()
                
            ...
        
        # 오디오 버퍼에 메트로놈 소리없이 녹음소리만 저장
        try:
            self.audio_buffer.put_nowait(indata.copy())
        except queue.Full:
            self.prtwl("Warning : Audio buffer is full! This block will be ignored.")
        except Exception as e:
            self.prtwl("Warning : ", e)
        


        

# 분석 스레드(threading으로 구성)
class AnalysisWorker:
    def __init__(self):
        ...
    

# 실행
if __name__ == "__main__":
    rpm_worker = RecordPlayMetronomeWorker()
    rpm_worker.thread.start()
    
    import time
    time.sleep(2)
    
    rpm_worker.thread.stop()
    rpm_worker.thread.close()