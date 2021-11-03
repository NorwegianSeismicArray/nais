"""
Author: Erik B. Myklebust, erik@norsar.no
2021
"""
import urllib3
import numpy as np

def spectrogram_standard_scaler(spectrograms):
    return spectrograms - spectrograms.mean(axis=0)[np.newaxis,:] / spectrograms.std(axis=0)[np.newaxis,:]

def spectrogram_minmax_scaler(spectrograms):
    return (spectrograms - spectrograms.min()) / (spectrograms.max() - spectrograms.min())

def waveform_minmax_scaler(waveforms):
    return (waveforms - waveforms.min()) / (waveforms.max() - waveforms.min())

def download_weights(url):

    file_name = url.split('/')[-1]
    u = urllib3.urlopen(url)
    f = open(file_name, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print("Downloading: %s Bytes: %s" % (file_name, file_size))

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8) * (len(status) + 1)
        print(status)

    f.close()

from zipfile import ZipFile

def extract_weights(filename, dest='models'):
    with ZipFile('filename', 'r') as zipObj:
       # Extract all the contents of zip file in current directory
       zipObj.extractall(dest)


