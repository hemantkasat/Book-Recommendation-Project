from collections import defaultdict
import json
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import svm
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
import networkx as nx
from sklearn import preprocessing
import sys
import csv

G = nx.Graph()
noreviews = []
avgrating = []
categoryset = defaultdict(set)


def initializegraph(category):

	global G
	global noreviews
	global avgrating
	global categoryset

	f = open('./Graphs/' + category + '/item_item.csv',"r")
	item_item = f.read()
	item_item = item_item.split('\n')
	del item_item[-1]

	for item in item_item:
		G.add_edge(item.split(' ')[0].strip(),item.split(' ')[1].strip())


	with open('./metadatabycategory/' + category + '.json',"r") as f:
		allbooks = json.load(f)


	with open('./Graphs/' + category + '/nodemap.json',"r") as f:
		allitemsmap = json.load(f)

	with open('./Dataset/ProductDetails.json',"r") as f:
		ProductDetail = json.load(f)



	# Bookdetails = {}

	documents = []
	
	nobooksnoreviews = 0

	for item in allbooks:
		it = item.split('\r\n')
		asin = it[1].split(':')[1].strip()
		# Bookdetails[allitemsmap[asin]] = item
		revs = item.split('\r\n')[2].split(':')[1].strip()
		if asin in ProductDetail.keys():
			revs += '\n'
			revs += '\n'.join(ProductDetail[asin])
		documents.append(revs)
		if item.split('\r\n')[3].split(':')[1].strip()!='Book':
			print item.split('\r\n')[3].split(':')[1].strip()

		for i in it:
			if 'reviews' in i:
				noreviews.append(i.strip().split('reviews:')[1].strip().split('total:')[1].strip().split('downloaded:')[0])
				avgrating.append(i.strip().split('avg rating:')[1].strip())
				# if(i.strip().split('avg rating:')[1].strip()=='0'):
					# print type(allitemsmap[asin])
					# if(allitemsmap[asin] in G.nodes()):
						# nobooksnoreviews = nobooksnoreviews + 1
			if '|Books' in i:
				for c in i.strip().split('|')[1:]:
					categoryset[allitemsmap[asin]].add(c)

	print documents[123]


	vectorizer  = TfidfVectorizer()
	tfidf = vectorizer.fit_transform(documents)
	tfidf = tfidf.toarray()
	return tfidf



def getnodepopularity(node1,node2):
	global G
	return (G.degree(node1) + G.degree(node2))

def getdistance(node1,node2):
	global G
	try:
		pathlength = nx.shortest_path_length(G,source=node1,target=node2)
	except:
		pathlength = 7
	return pathlength

def getcommonneighbour(node1,node2):
	global G
	nigh = nx.common_neighbors(G,node1,node2)
	return len(sorted(nigh))




def gettitlesimilarity(node1,node2,tfidf):
	return 1-spatial.distance.cosine(tfidf[int(node1)].tolist(),tfidf[int(node2)].tolist())


def getdiffreviews(node1,node2):
	global noreviews
	return abs((int(noreviews[int(node1)])-int(noreviews[int(node2)])))

def getcategorysimilarity(node1,node2):
	global categoryset
	return len(set.intersection(categoryset[int(node1)],categoryset[int(node2)]))

def getavgratingdifference(node1,node2):
	global avgrating
	return abs((float(avgrating[int(node1)])-float(avgrating[int(node2)])))




def createData(category):
	with open('./Graphs/' + category + '/ground_item_item.csv',"r") as f:
		reader = csv.reader(f)
		data = list(reader)
	
	X_data= []
	Y_data= []


	tfidf = initializegraph(category)


	for it in data:
		item = it[0].split(' ')
		nodepopularity = getnodepopularity(str(item[0]),str(item[1]))
		distanceingraph = getdistance(str(item[0]),str(item[1]))
		commonneighbour = getcommonneighbour(str(item[0]),str(item[1]))
		titlesimilarity = gettitlesimilarity(item[0],item[1],tfidf)
		noreviewssimilarity = getdiffreviews(item[0],item[1])
		categorysimilarity = getcategorysimilarity(item[0],item[1])
		avgratingdifference = getavgratingdifference(item[0],item[1])
		print nodepopularity,distanceingraph,commonneighbour,titlesimilarity,noreviewssimilarity,categorysimilarity,avgratingdifference
		X_data.append([nodepopularity,distanceingraph,commonneighbour,titlesimilarity,noreviewssimilarity,categorysimilarity,avgratingdifference])
		# X_data.append([titlesimilarity])
		Y_data.append(item[2])

	return X_data,Y_data






def svmmodel(X_data,Y_data):
	X_data = np.array(X_data)
	Y_data = np.array(Y_data)


	print X_data.shape
	print Y_data.shape

	min_max_scaler = preprocessing.MinMaxScaler()
	X_data = min_max_scaler.fit_transform(X_data)

	clf = svm.SVC(kernel='linear',class_weight='balanced',C=3)
	scores = cross_val_score(clf,X_data,Y_data,cv=5)
	return scores


if __name__ == '__main__':
	category = sys.argv[1]
	X_data,Y_data = createData(category)
	scores = svmmodel(X_data,Y_data)
	print scores
