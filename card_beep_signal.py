import numpy as np
import pylab
from scipy.io.wavfile import write
import os

Fs = 44100.0  # 샘플링 레이트
tlen = 1      # 재생시간 1초
Ts = 1 / Fs
t = np.arange(0, tlen, Ts)

sin_freq = 440
signal = np.sin(2 * np.pi * sin_freq * t)

noise = np.random.uniform(-1, 1, len(t)) * 0.1
signal_n = signal + noise

signal_f = np.fft.fft(signal_n)
freq = np.fft.fftfreq(len(t), Ts)

pylab.plot(freq, 20 * np.log10(np.abs(signal_f)))
pylab.xlim(0, Fs / 2)

scaled = np.int16(signal_n / np.max(np.abs(signal_n)) * 32767)
write('test.wav', 44100, scaled)

os.system("start test.wav")
