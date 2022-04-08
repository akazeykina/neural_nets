import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import activations
import numpy as np
import matplotlib.pyplot as plt
# Recommended way of RNG
from numpy.random import default_rng

def generate_train_data(
	target_function,
	interval_start,
	interval_end,
	size,
	rng,
	noise=0,
):
	''' 
	Returns the tuple (x, y) where x is the sequence of features
	and y is the sequence of labels.
	x is sampled from the uniform distribution on the interval,
	while y is optionally noised. 
	'''
	x = rng.uniform(interval_start, interval_end, size)
	y = target_function(x)
	if (noise != 0):
		y = y + rng.normal(0, noise, size)
	return (x, y)

def generate_validate_data(
	target_function,
	interval_start,
	interval_end,
	size=1000
):
	'''
	Returns the tuple (x, y) where x is the sequence of features
	and y is the sequence of labels. 
	'''
	x = np.linspace(interval_start, interval_end, size)
	y = target_function(x)
	return (x, y)

def plot_result(
	target_function,
	interval_start,
	interval_end,
	model,
	size=1000,
	train_data=None
):
	'''
	Plot the true and predicted value of the target function. 
	Optionally also plot the data used from training. 
	'''
	x, y_true = generate_validate_data(
		target_function,
		interval_start,
		interval_end,
		size)
	y_predict = model.predict(x)
	plt.plot(x, y_true)
	plt.plot(x, y_predict)
	if (train_data != None):
		plt.plot(train_data[0], train_data[1], "o")

if __name__ == "__main__":
	# Try a simple dataset
	# Sine function
	target_function  = np.sin
	interval_start   = 0.0
	interval_end     = 10.0
	train_data_noise = 0.1
	n_train          = 100
	n_val            = 1000
	epochs           = 500

	rng = default_rng()
	
	x_train, y_train = generate_train_data(
		target_function,
		interval_start,
		interval_end,
		n_train,
		rng,
		noise=train_data_noise)

	# Define the model
	# 3-layer fulled connected NN
	model = keras.Sequential([
	    keras.layers.Dense(32, activation=activations.relu, input_shape=[1]),
	    keras.layers.Dense(32, activation=activations.relu),
	    keras.layers.Dense(32, activation=activations.relu),
	    keras.layers.Dense(1)])

	# Print a summary including the number of parameters
	# in the NN
	#print(model.summary())

	# Define loss function and optimizer
	model.compile(
		loss='mean_squared_error',
		optimizer=keras.optimizers.RMSprop(0.0099))

	# Train
	model.fit(
		x_train,
		y_train,
		verbose=0,
		epochs=epochs)

	# Compare
	plot_result(
		target_function,
		interval_start,
		interval_end,
		model,
		train_data=(x_train, y_train))
	plt.show()
