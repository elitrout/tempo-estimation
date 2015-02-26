import essentia
import essentia.standard
import essentia.streaming
from essentia.standard import *
from pylab import plot, show, figure
import matplotlib.pyplot as plt
import numpy as np

eps = np.finfo(float).eps

fs = 44100.0  # ismir04 dataset
nfft = 2048
hopsize = 512

audiofile = '01-Dancing Queen.wav'
loader = essentia.standard.MonoLoader(filename = audiofile)
audio = loader()
# result of AudioLoader is significantly different, but audio == audio2 is True. Why?!
# loader2 = essentia.standard.AudioLoader(filename = audiofile)
# audio2, fs, nCh = loader2()
# audio2 = audio2[:, nCh - 1]
# audio = audio2

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
plt.figure(1)
plt.plot(energy)

# use EnergyBand in essentia. result is different
# energyB = np.zeros([nfr, bands])
# for fr in range(nfr):
#     for i in range(bands):
#         energyBand = EnergyBand(startCutoffFrequency=fco[i], stopCutoffFrequency=(1+fco[i+1]))  # stop freq included?
#         energyB[fr, i] = energyBand(specgram[fr])
# energy[energy < eps] = eps
# energyB = 10 * np.log10(energyB)
# plot(energyB)

corrtime = 6.0
nfr_corr = np.round(corrtime * fs / hopsize)
corr_matrix = np.zeros([nfr_corr, bands])
for nband in range(bands):
    e = energy[: nfr_corr, nband]
    x = np.correlate(e - np.mean(e), e - np.mean(e), mode='full')
    x = x / x[nfr_corr - 1]
    corr_matrix[:, nband] = x[nfr_corr - 1 : 2 * nfr_corr - 1]
# plot(corr_matrix)
# plot(x)

tt = np.arange(nfr_corr, dtype = np.float64) * hopsize / fs
tt[tt < eps] = eps
bpm = 60 / tt

plt.figure(2)
plt.subplot(211)
plt.plot(corr_matrix)
plt.subplot(212)
plt.plot(bpm, corr_matrix)
xlim(30, 240)
plt.show()

hop_corrtime = 1
hop_corr = np.round(hop_corrtime * fs / hopsize)
nfr_bpm = int(np.round((nfr - nfr_corr) / hop_corr))
bpm_matrix = np.zeros([nfr_corr, bands, nfr_bpm])

for k in range(nfr_bpm):
    corr_matrix = np.zeros([nfr_corr, bands])
    for nband in range(bands):
        b = k * hop_corr
        e = energy[b : b+nfr_corr, nband]
        x = np.correlate(e - np.mean(e), e - np.mean(e), mode='full')
        x = x / x[nfr_corr - 1]
        corr_matrix[:, nband] = x[nfr_corr - 1 : 2 * nfr_corr - 1]
    bpm_matrix[:, :, k] = corr_matrix

