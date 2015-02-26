import essentia
import essentia.standard
import essentia.streaming
from essentia.standard import *
from pylab import plot, show, figure
import matplotlib.pyplot as plt
import numpy as np
import os

# buggy
def peakDetection(arr, w):
    x = arr.copy()
    peaks = []
    for i in range(len(x)):
        if np.all(x[i] > x[max(0, i-w) : i]) and np.all(x[i] > x[i+1 : min(len(x), i+w+1)]):
            peaks.append(i)
    return peaks

def tempo(audiofile = './01-Dancing Queen.wav'):
    eps = np.finfo(float).eps
    
    fs = 44100.0  # ismir04 dataset
    nfft = 2048
    hopsize = 512
    bpm_boudaryL = 30
    bpm_boudaryH = 240
    nPeaks = 3
    
    # audiofile = '01-Dancing Queen.wav'
    loader = essentia.standard.MonoLoader(filename = audiofile)
    audio = loader()
    # result of AudioLoader is significantly different, but audio == audio2 is True. Why?!
    loader2 = essentia.standard.AudioLoader(filename = audiofile)
    audio2, fs, nCh = loader2()
    audio2 = audio2[:, nCh - 1]
    audio = audio2
    
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
    
    # compute energy
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
    
    # auto-correlation
    corrtime = 6.0
    nfr_corr = np.round(corrtime * fs / hopsize)
    corr_matrix = np.zeros([nfr_corr, bands])
    for nband in range(bands):
        e = energy[: nfr_corr, nband]
        x = np.correlate(e - np.mean(e), e - np.mean(e), mode='full')
        x = x / x[nfr_corr - 1]
        corr_matrix[:, nband] = x[nfr_corr - 1 : 2 * nfr_corr - 1]
    
    tt = np.arange(nfr_corr, dtype = np.float64) * hopsize / fs
    tt[tt < eps] = eps
    bpm = 60 / tt
    # index for restricted bpm
    idxH = np.where(bpm >= bpm_boudaryL)
    idxH = idxH[0][-1]
    idxL = np.where(bpm <= bpm_boudaryH)
    idxL = idxL[0][0]
    corr_matrixR = corr_matrix[idxL : idxH+1, :]
    bpm = bpm[bpm >= bpm_boudaryL]
    bpm = bpm[bpm <= bpm_boudaryH]
    
    plt.figure(2)
    plt.subplot(311)
    plt.plot(corr_matrix)
    plt.subplot(312)
    peaks = peakDetection(corr_matrixR[:, 0], 2)
    for i in range(len(peaks)):
        plt.axvline(bpm[peaks[i]], color='g')
    plt.xlim(30, 240)
    plt.plot(bpm, corr_matrixR)
    
    hop_corrtime = 1
    hop_corr = np.round(hop_corrtime * fs / hopsize)
    nfr_bpm = int(np.round((nfr - nfr_corr) / hop_corr))
    bpm_matrix = np.zeros([idxH-idxL+1, bands, nfr_bpm])
    
    # compute bpm-gram based on auto-correlation
    for k in range(nfr_bpm):
        corr_matrix = np.zeros([nfr_corr, bands])
        for nband in range(bands):
            b = k * hop_corr
            e = energy[b : b+nfr_corr, nband]
            x = np.correlate(e - np.mean(e), e - np.mean(e), mode='full')
            x = x / x[nfr_corr - 1]
            corr_matrix[:, nband] = x[nfr_corr - 1 : 2 * nfr_corr - 1]
            corr_matrixR = corr_matrix[idxL : idxH+1, :]
        bpm_matrix[:, :, k] = corr_matrixR
    
    # compute peaks of bpm-gram
    peaks = peakDetection(bpm_matrix[:, 0, 0], 2)
    Npeaks = np.argsort(bpm_matrix[peaks, 0, 0])
    Npeaks = Npeaks[-3 :]
    peaks = [peaks[i] for i in Npeaks]
    for i in range(len(peaks)):
        plt.axvline(bpm[peaks[i]], color='g')
    plt.xlim(30, 240)
    plt.plot(bpm, bpm_matrix[:, 0, 0])
    plt.show()
    
    # compute N highest peaks for each band in bpm-gram
    peaks = np.zeros([nPeaks, bands, nfr_bpm])  # peak position
    peak_matrix = np.zeros([nPeaks, bands, nfr_bpm])  # correlation at peak position
    for nband in range(bands):
        for k in range(nfr_bpm):
            p = peakDetection(bpm_matrix[:, nband, k], 2)
            # if peaks are less than 3, use the last ones instead
            if len(p) < 3:
                p = peaks[:, nband, k-1]
                p = p.tolist()
                p = [int(i) for i in p]
            Np = np.argsort(bpm_matrix[p, nband, k])
            Np = Np[-nPeaks :]
            p = [p[i] for i in Np]
            p = np.sort(p)
            peaks[:, nband, k] = p
            peak_matrix[:, nband, k] = bpm_matrix[p, nband, k]
    
    # for each peak position of each band, compute its standard deviation and median correlation value along the bpm-gram. buggy!!!
    p_std = np.zeros([nPeaks, bands])
    p_mcorr = np.zeros([nPeaks, bands])
    for nP in range(nPeaks):
        for nband in range(bands):
            p_std[nP, nband] = np.std(peak_matrix[nP, nband, :])
            p_mcorr[nP, nband] = np.median(peak_matrix[nP, nband, :])
    
    # find the peak position for each band that has the lowest standard deviation (stable)
    p_lstd = np.argmax(p_std, axis=0)
    
    # find the band that has the highest median correlation value with the peak position aquired above
    p_mcorr2 = np.zeros(bands)
    for nband in range(bands):
        p_mcorr2[nband] = p_mcorr[p_lstd[nband], nband]
    b_hmed = np.argmax(p_mcorr2)
    
    # compute the median peak position of that band
    p_med = np.median(peaks[p_lstd[b_hmed], b_hmed, :])
    
    # compute corresponding bpm
    bpm_final = bpm[p_med]
    
    return bpm_final

# evaluation
datafolder = '../DATASET'
annofolder = '../DATASET/Fabiens annotations'
filelist = os.listdir(datafolder)
result = []
resultD = []  # double tempo
resultH = []  # half tempo
resultDH = []  # double or half tempo
for wavfile in filelist:
    if wavfile[-4:] == '.wav':
        bpm = tempo(audiofile = datafolder + '/' + wavfile)
        annofile = annofolder + '/' + wavfile[: -4] + ' beat.bpm'
        f = open(annofile, 'r')
        anno = f.read()
        anno = anno[: -1]  # trim the space at the end
        anno = float(anno)
        f.close()
        if bpm <= anno + 0.1 * anno and bpm >= anno - 0.1 * anno:
            result.append(1)
        else:
            result.append(0)
        if bpm <= 2 * anno + 0.1 * 2 * anno and bpm >= 2 * anno - 0.1 * 2 * anno:
            resultD.append(1)
        else:
            resultD.append(0 or result[-1])
        if bpm <= 0.5 * anno + 0.1 * 0.5 * anno and bpm >= 0.5 * anno - 0.1 * 0.5 * anno:
            resultH.append(1)
        else:
            resultH.append(0 or result[-1])
        resultDH.append(result[-1] or resultD[-1] or resultH[-1])
        print result[-1], resultD[-1], resultH[-1], resultDH[-1], bpm, anno

acc = float(sum(result)) / len(result)
accD = float(sum(resultD)) / len(resultD)
accH = float(sum(resultH)) / len(resultH)
accDH = float(sum(resultDH)) / len(resultDH)
print acc, accD, accH, accDH
