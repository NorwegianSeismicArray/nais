"""
Author: Erik B. Myklebust, erik@norsar.no
2021
"""

def spectrogram_standard_scaler(spectrograms):
    return spectrograms - spectrograms.mean(axis=0)[np.newaxis,:] / spectrograms.std(axis=0)[np.newaxis,:]

def spectrogram_minmax_scaler(spectrograms):
    return (spectrograms - spectrograms.min()) / (spectrograms.max() - spectrograms.min())

def waveform_minmax_scaler(waveforms):
    return (waveforms - waveforms.min()) / (waveforms.max() - waveforms.min())