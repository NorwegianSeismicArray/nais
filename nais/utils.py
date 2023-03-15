"""
Author: Erik B. Myklebust, erik@norsar.no
2021
"""
import urllib3
import numpy as np
import tensorflow as tf
tfl = tf.keras.layers

HERMITE = [[1, 0, -3, 2], [0, 0, 3, -2], [0, 1, -2, 1], [0, 0, -1, 1]]
FORMAT = 'float32'

def crop_and_concat(x, y):
    to_crop = x.shape[1] - y.shape[1]
    if to_crop < 0:
        to_crop = abs(to_crop)
        of_start, of_end = to_crop // 2, to_crop // 2
        of_end += to_crop % 2
        y = tfl.Cropping1D((of_start, of_end))(y)
    elif to_crop > 0:
        of_start, of_end = to_crop // 2, to_crop // 2
        of_end += to_crop % 2
        y = tfl.ZeroPadding1D((of_start, of_end))(y)
    return tfl.concatenate([x,y])

def crop_and_add(x, y):
    to_crop = x.shape[1] - y.shape[1]
    if to_crop < 0:
        to_crop = abs(to_crop)
        of_start, of_end = to_crop // 2, to_crop // 2
        of_end += to_crop % 2
        y = tfl.Cropping1D((of_start, of_end))(y)
    elif to_crop > 0:
        of_start, of_end = to_crop // 2, to_crop // 2
        of_end += to_crop % 2
        y = tfl.ZeroPadding1D((of_start, of_end))(y)
    return x + y

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

def complex_hermite_interp(xi, x, m, p):
    """Complex interpolation with hermite polynomials.

    Arguments
    ---------
        x: array-like
            The knots onto the function is defined (derivative and
            antiderivative).
        t: array-like
            The points where the interpolation is required.
        m: array-like
            The complex values of amplitude onto knots.
        p: array-like
            The complex values of derivatives onto knots.

    Returns
    -------
        yi: array-like
            The interpolated complex-valued function.
    """
    # Hermite polynomial coefficients
    h = tf.constant(np.array(HERMITE).astype(FORMAT))

    # Concatenate coefficients onto shifted knots (1, n_knots - 1)
    # The knots are defined at each scales, so xx is (n_scales, n_knots - 1, 2)
    xx = tf.stack([x[:, :-1], x[:, 1:]], axis=2)

    # The concatenated coefficients are of shape (2, n_knots - 1, 2)
    mm = tf.stack([m[:, :-1], m[:, 1:]], axis=2)
    pp = tf.stack([p[:, :-1], p[:, 1:]], axis=2)

    # Define the full function y to interpolate (2, n_knots - 1, 4)
    # on the shifted knots
    y = tf.concat([mm, pp], axis=2)

    # Extract Hermite polynomial coefficients from y (n_knots - 1, 4)
    yh = tf.einsum('iab,bc->iac', y, h)

    # Extract normalized piecewise interpolation vector
    # (n_scales, n_knots - 1, n_interp)
    xi_ = tf.expand_dims(tf.expand_dims(xi, 0), 0)
    x0_ = tf.expand_dims(xx[:, :, 0], 2)
    x1_ = tf.expand_dims(xx[:, :, 1], 2)
    xn = (xi_ - x0_) / (x1_ - x0_)

    # Calculate powers of normalized interpolation vector
    mask = tf.logical_and(tf.greater_equal(xn, 0.), tf.less(xn, 1.))
    mask = tf.cast(mask, tf.float32)
    xp = tf.pow(tf.expand_dims(xn, -1), [0, 1, 2, 3])

    # Interpolate
    return tf.einsum('irf,srtf->ist', yh, xp * tf.expand_dims(mask, -1))


def real_hermite_interp(xi, x, m, p):
    """Real interpolation with hermite polynomials.

    Arguments
    ---------
        x: array-like
            The knots onto the function is defined (derivative and
            antiderivative).
        t: array-like
            The points where the interpolation is required.
        m: array-like
            The real values of amplitude onto knots.
        p: array-like
            The real values of derivatives onto knots.

    Returns
    -------
        yi: array-like
            The interpolated real-valued function.
    """
    # Hermite polynomial coefficients
    h = tf.constant(np.array(HERMITE).astype(FORMAT))

    # Concatenate coefficients onto shifted knots (1, n_knots - 1)
    # The knots are defined at each scales, so xx is (n_scales, n_knots - 1, 2)
    xx = tf.stack([x[:, :-1], x[:, 1:]], axis=2)

    # The concatenated coefficients are of shape (n_knots - 1, 2)
    mm = tf.stack([m[:-1], m[1:]], axis=1)
    pp = tf.stack([p[:-1], p[1:]], axis=1)

    # Define the full function y to interpolate (n_knots - 1, 4)
    # on the shifted knots
    y = tf.concat([mm, pp], axis=1)

    # Extract Hermite polynomial coefficients from y (n_knots - 1, 4)
    yh = tf.matmul(y, h)

    # Extract normalized piecewise interpolation vector
    # (n_scales, n_knots - 1, n_interp)
    xi_ = tf.expand_dims(tf.expand_dims(xi, 0), 0)
    x0_ = tf.expand_dims(xx[:, :, 0], 2)
    x1_ = tf.expand_dims(xx[:, :, 1], 2)
    xn = (xi_ - x0_) / (x1_ - x0_)

    # Calculate powers of normalized interpolation vector
    mask = tf.logical_and(tf.greater_equal(xn, 0.), tf.less(xn, 1.))
    mask = tf.cast(mask, tf.float32)
    xp = tf.pow(tf.expand_dims(xn, -1), [0, 1, 2, 3])

    # Interpolate
    return tf.einsum('rf,srtf->st', yh, xp * tf.expand_dims(mask, -1))