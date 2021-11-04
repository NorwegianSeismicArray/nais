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


      
def get_model_memory_usage(batch_size, model):
    import numpy as np
    try:
        from keras import backend as K
    except:
        from tensorflow.keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes
