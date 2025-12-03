import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

# 배음 만들기
sr = 44100
duration = 4
decay_rate = 1.1

def make_attack_sound(f0, duration=2, decay_rate=decay_rate, just_note=False):
    # Time array
    t = np.linspace(0., duration, 
                    int(sr * duration), 
                    endpoint=False)
    # Sounds
    y_f0 = 0.5 * np.cos(2 * np.pi * f0 * t) * np.exp(-decay_rate * t)
    y_h2 = 0.25 * np.cos(2 * np.pi * 2*f0 * t) * np.exp(-decay_rate * t)
    y_h3 = 0.125 * np.cos(2 * np.pi * 3*f0 * t) * np.exp(-decay_rate * t)
    y_h4 = 0.0625 * np.cos(2 * np.pi * 4*f0 * t) * np.exp(-decay_rate * t)

    if just_note == True:
        y = y_f0
    else:
        y = y_f0 + y_h2 + y_h3 + y_h4
    
    y = librosa.util.normalize(y)
    
    return y

# First attack
y_first_atk = make_attack_sound(f0=165, just_note=True)     # C3
y_second_atk = make_attack_sound(f0=587, just_note=True)    # E3
y_third_atk = make_attack_sound(f0=440, just_note=True)     # G3

y = np.concatenate([y_first_atk, y_second_atk, y_third_atk])

# sound = librosa.chirp(fmin=330, fmax=800, duration=5, sr=44100)
# print(sound.shape)

fig, ax = plt.subplots()
librosa.display.waveshow(y, sr=44100, ax=ax)
plt.show()

sf.write('sound.wav', y, samplerate=44100, format='WAV')