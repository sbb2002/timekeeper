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
