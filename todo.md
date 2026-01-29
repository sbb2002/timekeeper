2026-01-21  분석스레드로부터 onset, pitch, timing 받아오기
                ㄴ기존 플로터에 있는 것 떼어내야함
            받아와서 인디케이터에 업데이트
                ㄴ비트 인디케이터처럼 일정주기로 결과값 업데이트하기
            sustain은???

            1) Sustain measure dev.
                - Adaptive threshold(ML?) w/ Energy decay
            2) Timing measure dev.
                - Onset index --> Time 역산
                - Time vs Beat at time: Subnote step으로 time을 mod처리
                - 시간차(mod)가 step의 반 이하면 fast, 이상이면 slow
            3) 분석스레드 중 플로터에서 onset, pitch, timing, sustain 이관
                - return 값만 있으면 됨
            4) UI "DETECTION RESULT"에서 onset, pitch, timing, sustain 표시
                - 디자인 재설계
                - 업데이트

2026-01-23  너무 복잡해서 rhythm, tuner, sustain checker 모두 떼기로 했음.
            blocksize나 sustain 때문에 latency 제약이 너무 커지고 복잡해짐.
            그래서 지금은 rhythm metronome 먼저 만들기.

            1) Window를 이용해 onset으로부터 정박 타이밍과 시간차이 계산하는 클래스 만들기
                - 윈도우 사이즈 W 정하기
                - 매 블록마다 onset을 이 클래스의 queue에 집어넣기
                - W 크기만큼 큐에 가득차면 first onset을 추려내기 (정확도 모니터링 필)
                - first onset ~ 정박 간의 시간차이를 계산해서 판정 반환하기
                    * Temporal resolution (Perfect, Great, Good, Bad, Miss)
                        >> Pro lv.: ± 2, 5, 10, 20, 50
                        >> Beginner lv.: ± 25, 35, 50, 65, 80
                - 반환 끝났으면 이 큐를 모두 비우고 다시 onset 수집
                - Stop을 누를 경우 이 클래스도 GC할 것

            2) Onset Queue

![image](pictures\20260123_160302.jpg)
![image](pictures\20260123_160313.jpg)

2026-01-28  RhythmChecker 클래스를 만드는 중 난관에 부딪혔다.
            
            1) 인디케이터 업데이트 주기(1/24)와 박자 측정 주기(1 subnote)는 다르므로 이를 분리해야함
            2) 인디케이터 업데이트 주기에 따라 peak onset을 단 1번만 수집함.
                - 1)에서 말한 업데이트 주기와 연관된 것으로 생각
                - 저거 잡으면 해결될 것으로 예상

2026-01-29  이상하다. 첫번째 onset이 계속 무시되는데 잘보니 로깅이 이상하다.
            
            첫 onset은 86번 블록인데, 인풋은 1234부터 시작함.
            
            판정 타이밍(full)일 때마다 dq 열어보면 이상한 값들이 들어가있음.
            n_subnote에서 문제가 있음. 시작 1로 해야만 카운팅이 스타트되는데 
            첫 박자가 이니셜할 때 생성되어서 무조건 무시당함.

            # TODO
            [v]      가짜 온셋 생성 시 첫 박자(86번 블록 부근)에 생성안하는 문제
            [v]      판정할 때도 첫 박자는 무시됨(실제로는 4박자 쉬고 할거라 ㄱㅊ긴 함... 해결 후순위)

            이제 해결 후 테스트해봤더니 결과가 잘 나옴.
            ====== FINAL RESULTS ======
            {'delta_t': [-1.8367346938775508, 0.6802721088435374, 6.099773242630386, -2.993197278911565, 11.133786848072562, 2.0408163265306123, 1.655328798185941, 7.074829931972789, -7.823129251700681, -5.306122448979592, 5.918367346938775], 'grade': ['PERFECT', 'PERFECT', 'GOOD(SLOW)', 'GREAT(FAST)', 'BAD(SLOW)', 'GREAT(SLOW)', 'PERFECT', 'GOOD(SLOW)', 'GOOD(FAST)', 'GOOD(FAST)', 'GOOD(SLOW)'], 'timestamp': [0.24816326530612245, 0.5006802721088436, 0.7560997732426303, 0.9970068027210884, 1.2611337868480725, 1.5020408163265306, 1.751655328798186, 2.0070748299319727, 2.2421768707482994, 2.4946938775510206, 2.755918367346939], 'subnote': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}

            []      onset.py에다가 버전업하고 실제 테스트