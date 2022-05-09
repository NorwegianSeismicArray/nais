import tensorflow as tf
import numpy as np
import math
from scipy.signal import convolve, tukey

class AugmentWaveformSequence(tf.keras.utils.Sequence):
    """
    x_set :
        numpy array, 3D, (samples, length, channels)
    y_set :
        numpy array or list of numpy arrays.
    event_type :
        list, type of event, eg. earthquake.
    snr :
        nparray, signal-to-noise for each event.
    batch_size : int, default=32
    y_type: str, default='single'
        single value or waveform (eg. autoencoders, p/s picking etc).
        'single': single pick, y is single value
        'region': region of data, y is tuple of start and end
    norm_mode:
        str, max or std
    augmentation :
        bool
    add_event :
        float, stack events at prob.
    add_gap :
        float, mask data to zeros in period at prob.
    max_gap_size :
        float, max zeros gap in data. Proportion.
    coda_ration :
        float
    shift_event :
        float, move arrivals at prob.
    drop_channel :
        float, drop channel at prob.
    scale_amplitude :
        float, scale amplitude at prob.
    pre_emphasis :
        float
    min_snr :
        float, minimum snr required to perform augmentation.
    buffer :
        minimum steps from start of windown to p-arrival.
    shuffle :
        bool, shuffle the dataset on epoch end.
    """

    def __init__(self,
                 x_set,
                 y_set,
                 event_type,
                 snr=None,
                 batch_size=32,
                 y_type='single',
                 norm_mode='max',
                 norm_channel_mode='local',
                 augmentation=False,
                 ramp=0,
                 taper_alpha=0.0,
                 add_event=0.0,
                 add_gap=0.0,
                 max_gap_size=0.1,
                 coda_ratio=0.4,
                 new_length=None,
                 add_noise=0.0,
                 drop_channel=0.0,
                 scale_amplitude=0.0,
                 pre_emphasis=0.97,
                 min_snr=10.0,
                 buffer=0,
                 shuffle=False
                 ):
        self.x, self.y = x_set, y_set
        self.num_channels = self.x.shape[-1]
        self.y_type = y_type
        self.event_type = event_type
        if snr is None:
            snr = np.zeros(x_set.shape[0])
        self.snr = snr
        self.taper_alpha = taper_alpha

        if new_length is None:
            self.new_length = int(0.8 * self.x.shape[1])
        else:
            self.new_length = new_length

        if not (isinstance(self.y, list) or not isinstance(self.y, tuple)):
            self.y = [self.y]
            self.y_type = [self.y_type]

        self.batch_size = batch_size
        self.min_snr = min_snr
        self.shuffle = shuffle
        self.p_buffer = buffer
        self.norm_mode = norm_mode
        self.norm_channel_mode = norm_channel_mode
        self.augmentation = augmentation
        self.add_event = add_event
        self.add_gap = add_gap
        self.max_gap_size = max_gap_size
        self.coda_ratio = coda_ratio
        self.add_noise = add_noise
        self.drop_channel = drop_channel
        self.scale_amplitude = scale_amplitude
        self.pre_emphasis = pre_emphasis
        self.use_ramp = ramp > 0
        self.ramp = np.ones(ramp) if ramp > 0 else 0
        self.on_epoch_end()

    def __len__(self):
        if self.augmentation:
            return 2 * int(np.floor(len(self.event_type) / self.batch_size))
        else:
            return int(np.floor(len(self.event_type) / self.batch_size))

    def __getitem__(self, item):
        if self.augmentation:
            indexes = self.indexes[item * self.batch_size // 2:(item + 1) * self.batch_size // 2]
            indexes = np.append(indexes, indexes)
        else:
            indexes = self.indexes[item * self.batch_size:(item + 1) * self.batch_size]
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.x))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _normalize(self, X, mode='max', channel_mode='local'):
        X -= np.mean(X, axis=0, keepdims=True)

        if mode == 'max':
            if channel_mode == 'local':
                m = np.max(X, axis=0, keepdims=True)
            else:
                m = np.max(X, keepdims=True)
        elif mode == 'std':
            if channel_mode == 'local':
                m = np.std(X, axis=0, keepdims=True)
            else:
                m = np.std(X, keepdims=True)
        else:
            raise NotImplementedError(
                f'Not supported normalization mode: {mode}')

        m[m == 0] = 1
        return X / m

    def _scale_amplitute(self, X, rate=0.1):
        n = X.shape[-1]
        r = np.random.uniform(0, 1)
        if r < rate:
            X *= np.random.uniform(size=n)[np.newaxis]
        elif r < 2 * rate:
            X /= np.random.uniform(size=n)[np.newaxis]
        return X

    def _drop_channel(self, X, snr, rate):
        n = X.shape[-1]
        X = np.copy(X)
        if np.random.uniform(0, 1) < rate and snr >= self.min_snr:
            c = [np.random.choice([0, 1]) for _ in range(n)]
            if sum(c) > 0:
                X[..., np.array(c) == 0] = 0
        return X

    def _drop_channel_noise(self, X, rate):
        return self._drop_channel(X, float('inf'), rate)

    def _add_gaps(self, X, rate, max_size=0.1):
        X = np.copy(X)
        l = X.shape[0]
        gap_start = np.random.randint(0, int((1 - max_size) * l))
        gap_end = np.random.randint(gap_start, l)
        gap_end = int(min(gap_end, gap_end + max_size * l))
        if np.random.uniform(0, 1) < rate:
            X[gap_start:gap_end] = 0
        return X

    def _add_noise(self, X, snr, rate):
        if np.random.uniform(0, 1) < rate and snr >= self.min_snr:
            noisy_X = np.empty_like(X)
            for c in range(X.shape[-1]):
                noisy_X[:, c] = X[:, c] + np.random.normal(
                    0, np.random.uniform(0.01, 0.15) * max(X[:, c]), X.shape[0])
        else:
            noisy_X = X
        return noisy_X

    def _adjust_amplitute_for_multichannels(self, X):
        t = np.max(np.abs(X), axis=0, keepdims=True)
        nt = np.count_nonzeros(t)
        if nt > 0:
            X *= X.shape[-1] / nt
        return nt

    def _triangular_label(self, a=0, b=20, c=40):

        z = np.linspace(a, c, num=2 * (b - a) + 1)
        y = np.zeros_like(z)
        y[z <= a] = 0
        y[z >= c] = 0
        first_half = np.logical_and(a < z, z <= b)
        y[first_half] = (z[first_half] - a) / (b - a)
        second_half = np.logical_and(b < z, z < c)
        y[second_half] = (c - z[second_half]) / (c - b)
        return y

    def _add_event(self, X1, detection1, X2, detection2, snr, rate, space=10):
        # Add a second event into empty part of trace.
        start1, end1 = detection1
        start2, end2 = detection2
        event2 = X2[start2:end2]
        event2_size = end2 - start2
        r = np.random.uniform(0, 1)
        if r < rate and snr >= self.min_snr:
            scale = 1 / np.random.uniform(1, 10)
            before = np.random.choice([True, False])
            after = not before
            if event2_size < start1 - space and before:
                # before first event
                t = np.zeros_like(X1[:start1 - space])
                t[:len(event2)] = event2
                t = np.roll(t, np.random.randint(
                    0, len(t) - len(event2)), axis=0)
                X1[:len(t)] += t * scale
            elif event2_size > len(X1) - end1 + space and after:
                # after first event
                t = np.zeros_like(X1[end1 + space:])
                t[:len(event2)] = event2
                t = np.roll(t, np.random.randint(
                    0, len(t) - len(event2)), axis=0)
                X1[-len(t):] += t * scale

        return X1

    def _shift_crop(self, img, mask, detection):
        start, end = detection
        middle = start + (end - start) // 2
        y = np.random.randint(0, self.new_length) // 2
        s = int(middle - y)
        e = int(middle + (self.new_length-y))
        return img[s:e], mask[s:e]

    def _taper(self, img, mask, alpha=0.1):
        w = tukey(img.shape[0], alpha)
        return img*w[:,np.newaxis], mask

    def _pre_emphasis(self, X, pre_emphasis=0.97):
        for ch in range(X.shape[-1]):
            bpf = X[:, ch]
            X[:, ch] = np.append(bpf[0], bpf[1:] - pre_emphasis * bpf[:-1])
        return X

    def __convert_y_to_regions(self, y, yt, label):
        for j in range(len(y)):
            if yt[j] == 'single':
                i = int(y[j])
                if not math.isnan(i):
                    label[i,j] = 1
            elif yt[j] == 'region':
                start, end = y[j]
                if not math.isnan(start and end):
                    start, end = map(int, (start, end))
                    detection = (start, end)
                    label[start:end,j] = 1
            else:
                raise NotImplementedError(yt[j] + ' is not supported.')

            if self.use_ramp:
                label[:, j] = convolve(label[:, j], self.ramp, mode='same')

        if not 'detection' in locals():
            detection = (len(label)//4,3*len(label)//4)

        return label, detection

    def __data_generation(self, indexes):

        features = []
        labels = []

        for i, idx in enumerate(indexes):
            x = self.x[idx].copy()
            y = [a[idx].copy() for a in self.y]
            label = np.zeros((x.shape[0],len(self.y_type)))
            label, detection = self.__convert_y_to_regions(y, self.y_type, label)

            if self.augmentation and i > self.batch_size // 2:
                if self.event_type[i] == 'noise':
                    if self.drop_channel:
                        x = self._drop_channel_noise(x, self.drop_channel)
                    if self.add_gap:
                        x = self._add_gaps(
                            x, self.add_gap, max_size=self.max_gap_size)
                else:
                    if self.add_event:
                        t = np.random.choice(np.where(self.event_type != 'noise')[0])
                        _, detection2 = self.__convert_y_to_regions(0, t, label)
                        x = self._add_event(x, detection, self.x[t], detection2, self.snr[i], self.add_event)
                    if self.add_noise:
                        x = self._add_noise(x, self.snr[i], self.add_noise)
                    if self.drop_channel:
                        x = self._drop_channel(x, self.snr[i], self.drop_channel)
                    if self.scale_amplitude:
                        x = self._scale_amplitute(x, self.scale_amplitude)
                    if self.pre_emphasis:
                        x = self._pre_emphasis(x, self.pre_emphasis)

            if self.norm_mode is not None:
                x = self._normalize(x, mode=self.norm_mode, channel_mode=self.norm_channel_mode)

            x, label = self._shift_crop(x, label, detection)

            if self.taper_alpha > 0:
                x, label = self._taper(x, label, self.taper_alpha)

            if len(x) != 0 and len(label) != 0:
                features.append(x)
                labels.append(label)

        features, labels = map(lambda a: np.stack(a, axis=0), (features, labels))

        labels = np.split(labels, len(self.y_type), axis=-1)
        return features, labels
