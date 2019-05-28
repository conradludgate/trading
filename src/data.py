from typing import Dict, List, Tuple

import requests
from os import getenv
from json import loads as parse_json
from pathlib import Path
from time import time, sleep
import numpy as np
from math import log

def get_data(cryptos):
	to_time = int(time() // (1440 * 60)) * 1440 * 60

	data = {}

	for crypto in cryptos:
		array = []
		i = -1
		while True:
			new_data = open_or_download(crypto, to_time - i * 1440 * 60, 1440)
			if "Data" not in new_data or len(new_data["Data"]) == 0:
				break

			for (i, v) in enumerate(new_data["Data"][:-1]):
				# new_data["Data"][i] = (log(v["close"]) + 4) / 13
				new_data["Data"][i] = (new_data["Data"][i + 1]["close"] - v["close"]) / v["close"] * 1000

			array = new_data["Data"][:-1] + array

			i += 1

		# print(crypto, min(array), max(array))
		data[crypto] = array

	return collect(data, cryptos)

def open_or_download(crypto: str, to_time: int, limit: int) -> Dict[str, any]:
	filename = Path("data") / "{0}-{1}-{2}.json".format(crypto, to_time, limit)

	try:
		with open(filename) as f:
			return parse_json(f.read())
	except:
		sleep(0.5)
		r = requests.get("https://min-api.cryptocompare.com/data/histominute", params={
			"fsym": crypto,
			"tsym": "GBP",
			"limit": limit,
			"toTs": to_time
		})

		data = r.json()
		with open(filename, "w") as f:
			f.write(r.text)

		return data

def collect(data: Dict[str, List[Dict[str, any]]], cryptos: List[str]) -> np.ndarray:
	values = np.ndarray([0, len(cryptos)])
	for (i, v) in enumerate(data[cryptos[0]]):
		value = np.ndarray([len(cryptos)])
		value[0] = v

		for (k, w) in enumerate(cryptos[1:]):
			value[k + 1] = data[w][i]

		# if i == 2 and j == 4:
		# 	print(value)

		values = np.concatenate((values, value.reshape(1, len(cryptos))))
	return values