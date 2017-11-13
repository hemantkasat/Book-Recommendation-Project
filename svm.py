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



with open('groundtruthwithlabels.json',"r") as f:
	groundtruthwithlabels = json.load(f)

with open('Publishing & Books.json',"r") as f:
	allbooks = json.load(f)

with open('allitemsmap.json',"r") as f:
	allitemsmap = json.load(f)

with open('productreviewsmap.json',"r") as f:
	productreviewsmap = json.load(f)



Y_data = []
X_data = []


Bookdetails = {}

documents = []
noreviews = []
avgrating = []
categoryset = defaultdict(set)
nobooksnoreviews = 0

for item in allbooks:
	it = item.split('\r\n')
	asin = it[1].split(':')[1].strip()
	Bookdetails[allitemsmap[asin]] = item
	revs = item.split('\r\n')[2].split(':')[1].strip()
	if asin in productreviewsmap.keys():
		revs += '\n'
		revs += '\n'.join(productreviewsmap[asin])
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


def gettitlesimilarity(node1,node2):
	return 1-spatial.distance.cosine(tfidf[node1].tolist(),tfidf[node2].tolist())


def getdiffreviews(node1,node2):
	return abs((int(noreviews[node1])-int(noreviews[node2])))

def getcategorysimilarity(node1,node2):
	return len(set.intersection(categoryset[node1],categoryset[node2]))

def getavgratingdifference(node1,node2):
	return abs((float(avgrating[node1])-float(avgrating[node2])))

f = open('item_item.csv',"r")
item_item = f.read()
item_item = item_item.split('\n')
del item_item[-1]

G = nx.Graph()
for item in item_item:
	G.add_edge(item.split(' ')[0].strip(),item.split(' ')[1].strip())

def getnodepopularity(node1,node2):
	return (G.degree(node1) + G.degree(node2))

def getdistance(node1,node2):
	try:
		pathlength = nx.shortest_path_length(G,source=node1,target=node2)
	except:
		pathlength = 7
	return pathlength

def getcommonneighbour(node1,node2):
	nigh = nx.common_neighbors(G,node1,node2)
	return len(sorted(nigh))

def extractfeatures():
	for item in groundtruthwithlabels:
		nodepopularity = getnodepopularity(str(item[0]),str(item[1]))
		distanceingraph = getdistance(str(item[0]),str(item[1]))
		commonneighbour = getcommonneighbour(str(item[0]),str(item[1]))
		titlesimilarity = gettitlesimilarity(item[0],item[1])
		noreviewssimilarity = getdiffreviews(item[0],item[1])
		categorysimilarity = getcategorysimilarity(item[0],item[1])
		avgratingdifference = getavgratingdifference(item[0],item[1])
		print nodepopularity,distanceingraph,commonneighbour,titlesimilarity,noreviewssimilarity,categorysimilarity,avgratingdifference
		X_data.append([nodepopularity,distanceingraph,commonneighbour,titlesimilarity,noreviewssimilarity,categorysimilarity,avgratingdifference])
		# X_data.append([titlesimilarity])
		Y_data.append(item[2])






print len(groundtruthwithlabels)
extractfeatures()


X_data = np.array(X_data)
Y_data = np.array(Y_data)


print X_data.shape
print Y_data.shape


min_max_scaler = preprocessing.MinMaxScaler()
X_data = min_max_scaler.fit_transform(X_data)

clf = svm.SVC(kernel='linear',class_weight='balanced',C=3)
scores = cross_val_score(clf,X_data,Y_data,cv=5)
print scores


if __name__ == '__main__':
	category = 