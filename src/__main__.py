from dotenv import load_dotenv

from model import lstm_model, training_generator

import tensorflow as tf

def main():
	load_dotenv()

	# to_time = int(time() // (1440 * 60) - 7) * 1440 * 60

	# data = {}

	cryptos = ["BTC","LTC","ETH","EOS","BCH","XRP","TRX","BNB"]
	# for crypto in cryptos:
	# 	array = []
	# 	for i in range(8):
	# 		array.append(open_or_download(crypto, to_time + i * 1440 * 60, 1440))

	# 	data[crypto] = array

	# data = collect(data, cryptos)

	# cryptos = ["BTC", "XRP"]

	model = lstm_model(len(cryptos))

	model.fit_generator(
	    training_generator(cryptos),
	    steps_per_epoch=128,
	    epochs=20,
	    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="loss")]
	)

if __name__ == "__main__":
	main()