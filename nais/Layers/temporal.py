
import tensorflow as tf 
import tensorflow.keras.layers as tfl


class TemporalBlock(tf.keras.Model):
	def __init__(self, dilation_rate, nb_filters, kernel_size, 
				       padding, dropout_rate=0.0): 
		super(TemporalBlock, self).__init__()
		init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
		assert padding in ['causal', 'same']

		# block1
		self.conv1 = tfl.Conv1D(filters=nb_filters, kernel_size=kernel_size,
				                   dilation_rate=dilation_rate, padding=padding, kernel_initializer=init)
		self.batch1 = tfl.BatchNormalization(axis=-1)
		self.ac1 = tfl.Activation('relu')
		self.drop1 = tfl.Dropout(rate=dropout_rate)
		
		# block2
		self.conv2 = tfl.Conv1D(filters=nb_filters, kernel_size=kernel_size,
						           dilation_rate=dilation_rate, padding=padding, kernel_initializer=init)
		self.batch2 = tfl.BatchNormalization(axis=-1)		
		self.ac2 = tfl.Activation('relu')
		self.drop2 = tfl.Dropout(rate=dropout_rate)

		self.downsample = tfl.Conv1D(filters=nb_filters, kernel_size=1, 
									    padding='same', kernel_initializer=init)
		self.ac3 = tfl.Activation('relu')

	def call(self, x, training):
		prev_x = x
		x = self.conv1(x)
		x = self.batch1(x)
		x = self.ac1(x)
		x = self.drop1(x) if training else x

		x = self.conv2(x)
		x = self.batch2(x)
		x = self.ac2(x)
		x = self.drop2(x) if training else x

		if prev_x.shape[-1] != x.shape[-1]:    # match the dimention
			prev_x = self.downsample(prev_x)
		assert prev_x.shape == x.shape

		return self.ac3(prev_x + x)            # skip connection

class TemporalConvNet(tf.keras.Model):
    def __init__(self, num_channels, kernel_size=2, dropout=0.2):
    	# num_channels is a list contains hidden sizes of Conv1D
        super(TemporalConvNet, self).__init__()
        assert isinstance(num_channels, list)

        # 初始化 model
        model = tf.keras.Sequential()

        # The model contains "num_levels" TemporalBlock
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_rate = 2 ** i                  # exponential growth
            model.add(TemporalBlock(dilation_rate, num_channels[i], kernel_size, 
                      padding='causal', dropout_rate=dropout))
        self.network = model

    def call(self, x, training):
        return self.network(x, training=training)