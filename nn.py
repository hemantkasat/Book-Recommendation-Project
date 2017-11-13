from __future__ import print_function
from collections import defaultdict
import json
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import svm
import csv
import sys
import tensorflow as tf
import keras
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.datasets import mnist
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from sklearn.metrics import accuracy_score
from keras.layers import Dense, Dropout, Flatten, Activation
from six.moves import cPickle
from keras import backend as K

def createData(category):
	with open('./Graphs/' + category + '/ground_item_item.csv', "r") as f:
		reader = csv.reader(f)
		data = list(reader)
	
	X_data = []
	Y_data = []

	f = open('./Embeddings/' + category + '/items.embeddings')
	embeddings = f.read()
	embeddings = embeddings.split('\n')
	del embeddings[-1]
	del embeddings[0]
	itemembeddings = defaultdict(list)

	for ite in embeddings:
		itemembeddings[ite.split(' ')[0]] = ite.split(' ')[1:]

	for items in data:
		item = items[0].split(' ')
		node1 = itemembeddings[str(item[0])]
		node2 = itemembeddings[str(item[1])]
		X_data.append(node1+node2)
		Y_data.append(item[2])
			
	X_data = np.array(X_data)
	Y_data = np.array(Y_data)

	return X_data,Y_data

def add_relu(model): 
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.051))

def DefineModel(input_dim, output_dim): 
	model = Sequential()
	model.add(Dense(128, input_shape = (input_dim, )))
	add_relu(model)
	model.add(Dense(64))
	add_relu(model)
	model.add(Dense(32))
	add_relu(model)
	model.add(Dense(output_dim, activation = 'softmax'))
	return model

def getCrossvalidationscore(model, X_data,Y_data):
	model.compile(optimizer = 'rmsprop', 
				  loss = 'categorical_crossentropy', 
				  metrics = ['accuracy'])
	processed_Y_data = []
	for y in Y_data:
		l = [0, 0, 0]
		if (y == 0) : l[1] = 1
		elif (y == 1): l[0] = 1
		else: l[2] = 1
		processed_Y_data.append(l)
	processed_Y_data = np.asarray(processed_Y_data)
	model.fit(X_data, processed_Y_data, epochs = 5, batch_size = 32)

if __name__ == '__main__':
	category = sys.argv[1]
	X_data,Y_data  = createData(category)
	print(X_data.shape,Y_data.shape)
	Model = DefineModel(200, 3)
	scores = getCrossvalidationscore(Model, X_data,Y_data)
	print(scores)
