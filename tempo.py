import essentia
import essentia.standard
import essentia.streaming
from essentia.standard import *
from pylab import plot, show, figure
import numpy as np

eps = np.finfo(float).eps

fs = 44100  # ??
nfft = 2048
hopsize = 512

audiofile = '01-Dancing Queen.wav'
loader = essentia.standard.MonoLoader(filename = audiofile)
audio = loader()

w = Windowing(type = 'hann')
spectrum = Spectrum()
specgram = []

for frame in FrameGenerator(audio, frameSize = nfft, hopSize = hopsize):
    specgram.append(spectrum(w(frame)))

nfr = len(specgram)
bands = 8
lowband = 100.0

fco = lowband * (float(fs) / 2 / lowband) ** (np.arange(bands) / float((bands - 1))) / fs * nfft
fco = np.insert(fco, 0, 0)
fco = np.round(fco)

energy = np.zeros([nfr, bands])
for fr in range(nfr):
    for i in range(bands):
        energy[fr, i] = np.sum(specgram[fr][fco[i] : (1 + fco[i+1])] ** 2)
energy[energy < eps] = eps
energy = 10 * np.log10(energy)

plot(energy)
