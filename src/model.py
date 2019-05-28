from typing import Optional

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell

from data import get_data

import numpy as np

# def mean_squared_increase_error(y_true, y_pred):
# 	return tf.math.reduce_mean(tf.math.square(tf.math.divide(tf.math.subtract(y,y_),y)))

def lstm_model(num_cryptos, seq_len: int = 256, batch_size: Optional[int]=64, stateful: bool=True) -> tf.keras.Model:
	"""Trading model: predict the next value given the previous values. 
	Works on 8 symbols simultainiously
	BTC,LTC,ETH,EOS,BCH,XRP,TRX,BNB."""
	inputs = tf.keras.layers.Input(
		name='seed', shape=(seq_len, num_cryptos,), batch_size=batch_size, dtype=tf.float32)
	lstm = tf.keras.layers.LSTM(num_cryptos, stateful=stateful, return_sequences=True)(inputs)
	# output = tf.keras.layers.Dense(2, input_shape = (300 + len(string.punctuation),), activation="sigmoid")(lstm)
	
	model = tf.keras.Model(inputs=[inputs], outputs=[lstm])
	model.compile(
		# optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01),
		optimizer='adam',
		loss='mse',
		metrics=['mse'])
	
	return model

def training_generator(cryptos, seq_len=256, batch_size=64):
	"""A generator yields (source, target) arrays for training."""

	source = get_data(cryptos)

	tf.logging.info('Input data length %d', source.shape[0])
	while True:
		offsets = np.random.randint(0, len(source) - seq_len, batch_size)

		# Our model uses sparse crossentropy loss, but Keras requires labels
		# to have the same rank as the input logits.  We add an empty final
		# dimension to account for this.
		yield (
			## shape = (batch_size, seq_len, 300)
			np.stack([source[idx:idx + seq_len] for idx in offsets]),

			## shape = (batch_size, seq_len, 300, 1)
			# np.expand_dims(
			np.stack([source[idx + 1:idx + seq_len + 1] for idx in offsets]),
				# -1),
		)