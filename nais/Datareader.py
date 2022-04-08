import tensorflow as tf
import numpy as np


class AugmentWaveformSequence(tf.keras.utils.Sequence):
	"""
    x_set :
        numpy array, 3D, (samples, length, channels)
    y_set :
        numpy array or list of numpy arrays.
    batch_size : int, default=32
    y_type: str, default='single'
        single value or waveform (eg. autoencoders, p/s picking etc).
        'single': single pick, y is single value
        'region': region of data, y is tuple of start and end
    """

	def __init__(self,
				 x_set,
				 y_set,
				 event_type,
				 snr,
				 batch_size=32,
				 y_type='single',
				 norm_mode='max',
				 augmentation=False,
				 add_event=0.0,
				 add_gap=0.0,
				 max_gap_size=0.1,
				 coda_ratio=0.4,
				 shift_event=0.9,
				 add_noise=0.0,
				 drop_channel=0.0,
				 scale_amplitude=0.0,
				 pre_emphasis=True,
				 shuffle=False
				 ):
		self.x, self.y = x_set, y_set
		self.num_channels = self.x.shape[-1]
		self.y_type = y_type
		self.event_type = event_type
		self.snr = snr

		if not (isinstance(self.y, list) or isinstance(self.y, tuple)):
			self.y = [self.y]

		self.batch_size = batch_size
		self.shuffle = shuffle
		self.norm_mode = norm_mode
		self.augmentation = augmentation
		self.add_event = add_event
		self.add_gap = add_gap
		self.max_gap_size = max_gap_size
		self.coda_ratio = coda_ratio
		self.shift_event = shift_event
		self.add_noise = add_noise
		self.drop_channel = drop_channel
		self.scale_amplitude = scale_amplitude
		self.pre_emphasis = pre_emphasis
		self.on_epoch_end()

	def __len__(self):
		if self.augmentation:
			return 2 * int(np.floor(len(self.x) / self.batch_size))
		else:
			return int(np.floor(len(self.x) / self.batch_size))

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

	def _normalize(self, X, mode='max'):
		X -= np.mean(X, axis=0, keepdims=True)

		if mode == 'max':
			m = np.max(X, axis=0, keepdims=True)
		elif mode == 'std':
			m = np.std(X, axis=0, keepdims=True)
		else:
			raise NotImplementedError(
				f'Not supported normalization mode: {mode}')

		m[m == 0] = 1
		return X / m

	def _scale_amplitute(self, X, rate=0.1):
		n = X.shape[-1]
		r = np.random.uniform(0, 1)
		if r < rate:
			X *= np.random.uniform((1, n))
		elif r < 2 * rate:
			X /= np.random.uniform((1, n))
		return X

	def _drop_channel(self, X, snr, rate):
		n = X.shape[-1]
		X = np.copy(X)
		if np.random.uniform(0, 1) < rate and all(snr >= 10.0):
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
		if np.random.uniform(0, 1) < rate and all(snr >= 10.0):
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
		if r < rate and all(snr >= 10.0):
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

	def _shift_event(self, X, y, detection, rate):
		start, end = detection

		if np.random.uniform(0, 1) < rate:
			roll = np.random.randint(-start, len(X) - end)
			X = np.roll(X, roll, axis=0)
			y = [np.roll(a, roll, axis=0) for a in y]
		return X, y

	def _pre_emphasis(self, X, pre_emphasis=0.97):
		for ch in range(X.shape[-1]):
			bpf = X[:, ch]
			X[:, ch] = np.append(bpf[0], bpf[1:] - pre_emphasis * bpf[:-1])
		return X

	def __convert_y_to_regions(self, i, idx, labels):
		y = [a[idx] for a in self.y]
		label = [label[i] for label in labels]
		for j in range(len(y)):
			if self.y_type[j] == 'single':
				label[j][y[j]] = 1
			elif self.y_type == 'region':
				start, end = y[j][0], y[j][1]
				detection = (start, end)
				label[j][start:end] = 1
		return label, detection

	def __data_generation(self, indexes):

		features = np.zeros((self.batch_size, *self.x.shape[1:]))
		labels = [np.zeros((self.batch_size, *self.x.shape[0], 1))
				  for _ in self.y]

		for i, idx in enumerate(range(indexes)):
			x = self.x[idx]
			label, detection = self.__convert_y_to_regions(i, idx, labels)

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
						_, detection2 = self.__convert_y_to_regions(0, t, labels)
						x = self._add_event(x, detection, self.x[t], detection2, self.snr[i], self.add_event)
					if self.add_noise:
						x = self._add_noise(x, self.snr[i], self.add_noise)
					if self.drop_channel:
						x = self._drop_channel(x, self.snr[i], self.drop_channel)
					if self.scale_amplitude:
						x = self._scale_amplitute(x, self.scale_amplitude)
					if self.pre_emphasis:
						x = self._pre_emphasis(x, self.pre_emphasis)

			if self.event_type[i] != 'noise':
				x, y = self._shift_event(x, label, detection, self.shift_event / 2)

			if self.norm_mode is not None:
				x = self._normalize(x, mode=self.norm_mode)

			features[i] = x
			labels[i] = label

		return features, labels


