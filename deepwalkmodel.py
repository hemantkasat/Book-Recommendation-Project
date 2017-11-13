from collections import defaultdict
import json
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import svm
import sys
import csv


def createData(category):
	with open('./Graphs/' + category + '/ground_item_item.csv',"r") as f:
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

def getCrossvalidationscore(X_data,Y_data):
	clf = svm.SVC(kernel='linear',class_weight='balanced',C=3)
	return cross_val_score(clf,X_data,Y_data,cv=5)


if __name__ == '__main__':
	category = sys.argv[1]
	X_data,Y_data  = createData(category)
	print X_data.shape,Y_data.shape
	scores = getCrossvalidationscore(X_data,Y_data)
	print scores
