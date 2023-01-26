import tensorflow as tf
import numpy as np
import math
from scipy.signal import convolve, tukey, triang, fftconvolve, oaconvolve
from scipy.signal.windows import gaussian

from scipy.ndimage import gaussian_filter1d

class AugmentWaveformSequence(tf.keras.utils.Sequence):

    def __init__(self,
                 x_set,
                 y_set,
                 event_type,
                 detection=None,
                 snr=None,
                 ids=None,
                 metadata_df=None,
                 metadata_cols=None,
                 batch_size=32,
                 y_type='single',
                 norm_mode='max',
                 norm_channel_mode='local',
                 augmentation=False,
                 ramp=0,
                 fill_value=0.0,
                 taper_alpha=0.0,
                 add_event=0.0,
                 add_gap=0.0,
                 max_gap_size=0.1,
                 coda_ratio=0.4,
                 new_length=None,
                 add_noise=0.0,
                 drop_channel=0.0,
                 scale_amplitude=0.0,
                 pre_emphasis=0.0,
                 min_snr=10.0,
                 add_event_space=40,
                 buffer=0,
                 model_type='phasenet',
                 shuffle=False,
                 random_crop=True,
                 create_label=False
                 ):
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
        self.x, self.y = x_set, y_set
        self.detection = detection
        self.num_channels = self.x.shape[-1]
        self.y_type = y_type
        self.event_type = event_type
        self.model_type = model_type
        if snr is None:
            snr = np.zeros(x_set.shape[0])
        self.snr = snr
        self.taper_alpha = taper_alpha
        self.fill_value = fill_value

        self.ids = ids
        self.metadata_df = metadata_df
        self.metadata_cols = metadata_cols

        if new_length is None:
            self.new_length = int(0.8 * self.x.shape[1])
        else:
            self.new_length = new_length

        if not (isinstance(self.y, list) or not isinstance(self.y, tuple)):
            self.y = [self.y]
            self.y_type = [self.y_type]

        self.detection_index = -1
        self.phase_index = np.arange(len(y_set))

        self.random_crop = random_crop
        self.add_event_space = add_event_space
        self.norm_mode = norm_mode
        self.norm_channel_mode = norm_channel_mode
        self.batch_size = batch_size
        self.min_snr = min_snr
        self.shuffle = shuffle
        self.p_buffer = buffer
        self.augmentation = augmentation
        self.add_event = add_event
        self.add_gap = add_gap
        self.max_gap_size = max_gap_size
        self.coda_ratio = coda_ratio
        self.add_noise = add_noise
        self.drop_channel = drop_channel
        self.scale_amplitude = scale_amplitude
        self.pre_emphasis = pre_emphasis
        self.ramp = ramp
        self.create_label = create_label
        self.non_noise_events = np.where(self.event_type != 'noise')[0]

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.event_type) / self.batch_size))

    def __getitem__(self, item):
        indexes = self.indexes[item * self.batch_size:(item + 1) * self.batch_size]
        X, y = zip(*list(map(self.data_generation, indexes)))
        y = np.stack(y, axis=0)
        y = np.split(y, y.shape[-1], axis=-1)
        if not self.metadata_df is None:
            m = [self.metadata_df.loc[i, self.metadata_cols].values.astype('float') if i in self.metadata_df.index else np.ones(len(self.metadata_cols))*self.fill_value for i in self.ids[indexes]]
            return np.stack(X, axis=0), y, np.stack(m, axis=0).reshape((-1, len(self.metadata_cols)))
        else:
            return np.stack(X, axis=0), y

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
        if np.random.uniform(0, 1) < rate and snr >= self.min_snr:
            c = np.random.randint(0,n)
            X[..., c] = 0
        return X

    def _drop_channel_noise(self, X, rate):
        return self._drop_channel(X, float('inf'), rate)

    def _add_gaps(self, X, rate, max_size=0.1):
        l = X.shape[0]
        if np.random.uniform(0, 1) < rate:
            gap_start = np.random.randint(0, int((1 - max_size) * l))
            gap_end = np.random.randint(gap_start, gap_start + int(max_size * l))
            X[gap_start:gap_end] = 0
        return X

    def _add_noise(self, X, snr, rate):
        if np.random.uniform(0, 1) < rate and snr >= self.min_snr:
            N = np.stack([np.random.normal(loc=np.random.uniform(0.01, 0.15) * m, size=X.shape[0]) for m in X.max(axis=0)], axis=-1)
            X += N
        return X

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
        scale = 0
        if r < rate and snr >= self.min_snr:
            scale = 1 / np.random.uniform(1, 10)
            before = np.random.choice([True, False])
            after = not before

            space_before = start1 - space
            space_after = len(X1) - end1 - space

            if event2_size < space_before and before:
                # before first event
                left_over = space_before - event2_size
                if left_over - space > 0:
                    s = np.random.randint(0, left_over-space)
                    e = s + len(event2)
                    X1[s:e] += event2 * scale

            elif event2_size < space_after and after:
                # after first event
                left_over = space_after - event2_size
                if left_over > space:
                    s = np.random.randint(space, left_over)
                    e = s + len(event2)
                    X1[end1+s:end1+e] += event2 * scale

        return X1, scale

    def _shift_crop(self, img, mask):
        if self.random_crop:
            y1 = np.random.randint(0, len(img) - self.new_length)
        else:
            y1 = int((len(img) - self.new_length) / 2)

        img = img[y1:y1 + self.new_length]
        mask = mask[y1:y1 + self.new_length]
        return img, mask

    def _taper(self, img, mask, alpha=0.1):
        w = tukey(img.shape[0], alpha)
        return img*w[:,np.newaxis], mask

    def _pre_emphasis(self, X, pre_emphasis=0.97):
        return np.stack([np.append(X[0,c], X[1:,c] - pre_emphasis * X[:-1,c]) for c in range(X.shape[1])], axis=-1)

    def _convert_y_to_regions(self, y, yt, label):
        for j in range(len(y)):
            if yt[j] == 'single':
                i = y[j]
                if not math.isnan(i):
                    label[int(i),j] = 1
            elif yt[j] == 'region':
                start, end = y[j]
                if not (math.isnan(start) or math.isnan(end)):
                    start, end = map(int, (start, end))
                    start = max(0, start)
                    end = min(len(label), end)
                    detection = (start, end)
                    label[start:end,j] = 1
            else:
                raise NotImplementedError(yt[j] + ' is not supported.')

        if self.ramp > 0:
            label = gaussian_filter1d(label, sigma=self.ramp, axis=0)

        m = np.amax(label, axis=0, keepdims=True)
        m[m == 0] = 1
        label /= m

        if not 'detection' in locals():
            detection = (len(label)//4,3*len(label)//4)

        return label, detection

    def data_generation(self, idx):

        x = self.x[idx]
        y = [a[idx] for a in self.y]
        if self.create_label:
            label = np.zeros((x.shape[0],len(self.y_type)))
            label, detection = self._convert_y_to_regions(y, self.y_type, label)
        else:
            detection = self.detection[idx]
            label = np.concatenate([np.expand_dims(detection, axis=-1), np.stack(y, axis=-1)], axis=-1)

        do_aug = self.augmentation and np.random.random() > 0.5
        if do_aug:
            if self.event_type[idx] == 'noise':
                if self.drop_channel > 0:
                    x = self._drop_channel_noise(x, self.drop_channel)
                if self.add_gap > 0:
                    x = self._add_gaps(
                        x, self.add_gap, max_size=self.max_gap_size)
            else:
                if self.add_event > np.random.uniform(0,1) and self.snr[idx] >= self.min_snr:
                    t = np.random.choice(self.non_noise_events)
                    y2 = [a[t] for a in self.y]
                    if self.create_label:
                        label2 = np.zeros((x.shape[0], len(self.y_type)))
                        label2, detection2 = self._convert_y_to_regions(y2, self.y_type, label2)
                    else:
                        detection2 = self.detection[t]
                        label2 = np.concatenate([np.expand_dims(detection2, axis=-1), np.stack(y2, axis=-1)], axis=-1)
                    roll = np.random.randint(0, label.shape[0])
                    label = np.roll(label, roll, axis=0)
                    x2 = np.roll(self.x[t], roll, axis=0)
                    scale = 1 / np.random.uniform(1, 10)
                    label = np.amax([label, label2 * scale], axis=0)
                    x = x + scale * x2

                if self.add_noise > 0:
                    x = self._add_noise(x, self.snr[idx], self.add_noise)
                if self.drop_channel > 0:
                    x = self._drop_channel(x, self.snr[idx], self.drop_channel)
                if self.scale_amplitude > 0:
                    x = self._scale_amplitute(x, self.scale_amplitude)
                if self.pre_emphasis > 0:
                    x = self._pre_emphasis(x, self.pre_emphasis)
                if self.add_gap > 0:
                    x = self._add_gaps(
                        x, self.add_gap, max_size=self.max_gap_size)

        x, label = self._shift_crop(x, label)
        if self.taper_alpha > 0:
            x, label = self._taper(x, label, self.taper_alpha)

        if self.norm_mode is not None:
            x = self._normalize(x, mode=self.norm_mode, channel_mode=self.norm_channel_mode)

        return x, label
