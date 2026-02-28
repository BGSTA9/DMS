python -c "
import numpy as np
import wave, struct, os
os.makedirs('assets', exist_ok=True)
sr, dur, freq = 44100, 1.5, 880
t = np.linspace(0, dur, int(sr*dur))
wave_data = (np.sin(2*np.pi*freq*t) * 0.6 * 32767 * (1 - t/dur)).astype(np.int16)
with wave.open('assets/alarm.wav', 'w') as f:
    f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
    f.writeframes(struct.pack('<' + 'h'*len(wave_data), *wave_data))
print('alarm.wav generated.')
"