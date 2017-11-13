import tensorflow as tf
import sys
import json
import csv
import gensim
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from imblearn.under_sampling import ClusterCentroids
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import confusion_matrix


with open('./Dataset/ProductDetails.json',"r") as f:
	ProductDetail = json.load(f)


word_dimension = 100
class_1 = 0
class_0 = 0
class_2 = 0

def getData(category):
	max_length = 0
	class_1 = 0
	class_0 = 0
	class_2 = 0

	with open('./Graphs/' + category + '/ground_item_item.csv',"r") as f:
		reader = csv.reader(f)
		data = list(reader)

	with open('./Graphs/' + category + '/' + 'nodemap.json',"r") as f:
		nodemap = json.load(f)


	X_data = []
	Y_data = []
	sentences = []
	for item in data:
		it = item[0].split(' ')
		invertednodemap = dict([[v,k] for k,v in nodemap.items()])
		st1 = ProductDetail[invertednodemap[int(it[0])]]['title']
		st2 = ProductDetail[invertednodemap[int(it[1])]]['title']
		try:
			st1 += ProductDetail[invertednodemap[int(it[0])]]['description']
		except:
			pass
		try:
			st2 += ProductDetail[invertednodemap[int(it[1])]]['description']

		except:
			pass

		sentences.append(st1.split(' '))
		sentences.append(st2.split(' '))
		if(len(st1.split(' '))>max_length):
			max_length = len(st1.split(' '))
		if(len(st2.split(' '))>max_length):
			max_length = len(st2.split(' '))
		X_data.append([st1,st2])
		Y_data.append(it[2])
		if(it[2]=="1"):
			class_1 = class_1 + 1
		if(it[2]=="0"):
			class_0 = class_0 + 1
		if(it[2]=="-1"):
			class_2 = class_2 + 1

	model = gensim.models.Word2Vec(sentences, min_count=1,workers=5,size=word_dimension)

	# print class_0,class_1,class_2

	return X_data,Y_data,model.wv,max_length,len(X_data)/float(class_0),len(X_data)/float(class_1),len(X_data)/float(class_2)


def last_relevant(output,length):
	batch_size = tf.shape(output)[0]
	max_length = tf.shape(output)[1]
	out_size = int(output.get_shape()[2])
	index = tf.range(0, batch_size) * max_length + (length - 1)
	flat = tf.reshape(output, [-1, out_size])
	relevant = tf.gather(flat, index)
	return relevant

def length(sequence):
	used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
	length = tf.reduce_sum(used, 1)
	length = tf.cast(length, tf.int32)
	return length

if __name__ == '__main__':
	category = sys.argv[1]
	X_data,Y_data,word_vectors,max_length,class_0,class_1,class_2 = getData(category)
	
	print len(X_data)

	rus = RandomOverSampler(random_state=0)
	
	X_data_resampled,Y_data_resampled = rus.fit_sample(X_data,Y_data)

	print(sorted(Counter(Y_data_resampled).items()))


	print X_data_resampled[:10]

	frame_size = 100
	num_hidden = 100


	# print class_0,class_1,class_2


	sum = class_2 + class_0 + class_1

	print class_1/sum,class_0/sum,class_2/sum

	class_weight = tf.constant([class_1/sum,class_0/sum,class_2/sum],dtype=tf.float32)
	sequence = tf.placeholder(tf.float32,[None,max_length, frame_size],name='seq')
	output, state = tf.nn.dynamic_rnn(tf.contrib.rnn.GRUCell(num_hidden),sequence,dtype=tf.float32,sequence_length=length(sequence),)
	last = last_relevant(output,length(sequence))
	features = [(last[0]+last[1])/2]
	num_classes = 3
	target = tf.placeholder(tf.float32,[None,num_classes])
	w = tf.reduce_sum(class_weight*target)

	weight = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1))
	bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
	prediction = tf.nn.softmax(tf.matmul(features, weight) + bias)

	# unweighted_loss = -tf.reduce_sum(target * tf.log(prediction))
	# cross_entropy = tf.reduce_mean(unweighted_loss * w)
	cross_entropy = -tf.reduce_sum(target * tf.log(prediction))

	optimizer = tf.train.AdamOptimizer()
	minimize = optimizer.minimize(cross_entropy)



	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(len(X_data_resampled)):
			input_data = np.zeros((2,max_length,frame_size))
			words = X_data_resampled[i][0].split(' ')
			for j in range(len(words)):
				wv = word_vectors[words[j]]
				input_data[0][j] = wv

			words = X_data_resampled[i][1].split(' ')
			for j in range(len(words)):
				wv = word_vectors[words[j]]
				input_data[1][j] = wv
			tar  = np.zeros((1,3))
			if(Y_data_resampled[i]==1):
				tar[0][0] = 1
			if(Y_data_resampled[i]==0):
				tar[0][1] = 1
			if(Y_data_resampled[i]==-1):
				tar[0][2] = 1
			p = sess.run([minimize,cross_entropy,prediction],feed_dict={sequence:input_data,target:tar})
			print p

		correct_classification = 0
		y_pred = []
		for i in range(len(X_data)):
			input_data = np.zeros((2,max_length,frame_size))
			words = X_data[i][0].split(' ')
			for j in range(len(words)):
				wv = word_vectors[words[j]]
				input_data[0][j] = wv

			words = X_data[i][1].split(' ')
			for j in range(len(words)):
				wv = word_vectors[words[j]]
				input_data[1][j] = wv
			tar  = np.zeros((1,3))
			if(Y_data[i]==1):
				tar[0][0] = 1
			if(Y_data[i]==0):
				tar[0][1] = 1
			if(Y_data[i]==-1):
				tar[0][2] = 1
			p = sess.run([prediction],feed_dict={sequence:input_data,target:tar})
			cl = np.argmax(p[0])
			# y_pred.append(cl)
			print p
			if(cl==0):
				y_pred.append("1")
			if(cl==1):
				y_pred.append("0")
			if(cl==2):
				y_pred.append("-1")
			if(Y_data[i]=="1"):
				if(cl==0):
					print "Correctly classified label= 1"
					correct_classification = correct_classification + 1
			if(Y_data[i]=="0"):
				if(cl==1):
					print "Correctly classified label= 0"
					correct_classification = correct_classification + 1
			if(Y_data[i]=="-1"):
				if(cl==2):
					print "Correctly classified label= -1"
					correct_classification = correct_classification + 1

		print "Accuracy is ",float(correct_classification)/len(X_data)
		C = confusion_matrix(Y_data, y_pred,labels=["1","0","-1"])
		print C


	# with tf.Session() as sess:
	# 	sess.run(tf.global_variables_initializer())
	# 	kfold = KFold(n_splits=5)

	# 	for train_indices, test_indices in kfold.split(X_data):
	# 		print "training= ",len(train_indices), " test= ",len(test_indices)
	# 		for i in train_indices:
	# 			input_data = np.zeros((2,max_length,frame_size))
	# 			words = X_data[i][0].split(' ')	
	# 			for i in range(len(words)):
	# 				wv = word_vectors[words[i]]
	# 				input_data[0][i] = wv

	# 			words = X_data[i][1].split(' ')
	# 			for i in range(len(words)):
	# 				wv = word_vectors[words[i]]
	# 				input_data[1][i] = wv

	# 			tar = np.zeros((1,3))
	# 			if(Y_data[i]==1):
	# 				tar[0][0] = 1
	# 			elif(Y_data[i]==0):
	# 				tar[0][1] = 1
	# 			else:
	# 				tar[0][2] = 1
	# 			p = sess.run([minimize,cross_entropy,prediction],feed_dict={sequence:input_data,target:tar})
	# 			print p


	# 		correct_classification = 0
	# 		for i in test_indices:
	# 			input_data = np.zeros((2,max_length,frame_size))
	# 			words = X_data[i][0].split(' ')	
	# 			for i in range(len(words)):
	# 				wv = word_vectors[words[i]]
	# 				input_data[0][i] = wv

	# 			words = X_data[i][1].split(' ')
	# 			for i in range(len(words)):
	# 				wv = word_vectors[words[i]]
	# 				input_data[1][i] = wv				
			
	# 			p = sess.run([prediction],feed_dict={sequence:input_data})
	# 			cl = np.argmax(p[0])
	# 			print p
	# 			print Y_data[i]
	# 			print cl
	# 			if(Y_data[i]=="1"):
	# 				if(cl==0):
	# 					correct_classification = correct_classification + 1
	# 			if(Y_data[i]=="0"):
	# 				if(cl==1):
	# 					print "asdfjsakl;fjsdajklfjkljaslkjfljlajlkfjlasjfjljsef"
	# 					correct_classification = correct_classification + 1
	# 			if(Y_data[i]=="-1"):
	# 				if(cl==2):
	# 					correct_classification = correct_classification + 1

	# 		print correct_classification,len(test_indices)

	# 		print "Accuracy is ", float(correct_classification)/len(test_indices)


