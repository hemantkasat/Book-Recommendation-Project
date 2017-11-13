import json
import itertools
from collections import defaultdict
import networkx as nx
import csv
import sys
import os
import shutil
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial
import random


def gettitlesimilarity(tfidf,node1,node2):
    return 1-spatial.distance.cosine(tfidf[node1].tolist(),tfidf[node2].tolist())


def appendtofile(filename,category,list):
	with open('./Graphs/'  + category + '/'  + filename,"a") as f:
		writer = csv.writer(f,delimiter=' ')
		writer.writerows(list)

def getGraphs(category):
	
	with open('./metadatabycategory/' + category + '.json') as f:
		Books = json.load(f)

	with open('./ProductDetails/ProductDetails.json',"r") as f:
		ProductDetail = json.load(f)


	cust_item = defaultdict(list)
	item_simitem = defaultdict(list)

	itemgraphnodes = set()
	grounditemgraphnodes = set()
	nodemap = {}
	noitems = 0
	Descriptions = []
	titles = []

	for items in Books:
		it = items.split('\r\n')
		for i in it:
			if "ASIN:" in i:
				item =  i.split(':')[1].strip()
			
				if item not in nodemap.keys():
					nodemap[item] = noitems
					noitems = noitems + 1
			if "title:" in i:
			    title = i.strip().split('title:')[1].strip()
			    titles.append(title)
			    try:
			    	des = ProductDetail[item]['description']
			    	string = title + '\n' + des
			    	Descriptions.append(string)
			    except:
			    	Descriptions.append(title)

			if "similar" in i:
				simitem = i.strip().split('similar:')[1].strip().split('  ')[1:]
				item_simitem[item] = simitem

			if "cutomer" in i:
				cust_item[i.split('cutomer:')[1].split('rating')[0].strip()].append(item)

	item_item = set()
	ground_item_item = set()

	vectorizer  = TfidfVectorizer()
	tfidf = vectorizer.fit_transform(Descriptions)
	tfidf = tfidf.toarray()
	


	for items in Books:
		try:
			asin = items.split('\r\n')[1].strip().split(':')[1].strip()
			buyafterviewing = ProductDetail[asin]['related']['also_bought']
			buyafterviewing.append(asin)
			a = itertools.combinations(buyafterviewing,2)
			for x in a:			
				try:
					node1 = nodemap[x[0]]
					node2 = nodemap[x[1]]
					similarity = gettitlesimilarity(tfidf,node1,node2)
					print len(item_item),similarity
					# print node1,node2
					if(node1>node2):
						# print "sakjdf;kaljlfjjafjkj;jfakf;jlk"
						item_item.add((node1,node2))
						# print len(item_item)
					else:
						item_item.add((node2,node1))
					itemgraphnodes.add(node1)
					itemgraphnodes.add(node2)
				except:
					pass 
		except:
			pass
	
	count = 0




	print len(item_item)

	with open('./Graphs/'  + category + '/item_item.csv',"w") as f:
		writer = csv.writer(f,delimiter=' ')
		writer.writerows(list(item_item))


	# with open('./Graphs/'  + category + '/item_item.csv',"a") as f:
	# 	writer = csv.writer(f,delimiter=' ')
	# 	for key in cust_item.keys():
	# 		itemlist = cust_item[key]
	# 		if len(itemlist) > 50:
	# 			itemlist = random.sample(itemlist,50)
	# 		a = itertools.combinations(itemlist,2)			
	# 		for x in a:
	# 			similarity = gettitlesimilarity(tfidf,nodemap[x[0]],nodemap[x[1]])
	# 			if((similarity>0.5)&(similarity<0.9)):
	# 				if(nodemap[x[0]]!=nodemap[x[1]]):
	# 					print titles[nodemap[x[0]]],titles[nodemap[x[1]]],'\n',similarity,'\n'
	# 					if(nodemap[x[0]]>nodemap[x[1]]):
	# 						writer.writerow([nodemap[x[0]],nodemap[x[1]]])
	# 					else:	
	# 						writer.writerow([nodemap[x[1]],nodemap[x[0]]])
	# 			        itemgraphnodes.add(nodemap[x[0]])
	# 			        itemgraphnodes.add(nodemap[x[1]])
	# 	print len(itemgraphnodes)

	G = nx.Graph()
	for key in item_simitem.keys():
		for it in item_simitem[key]:
			if(it in nodemap.keys()):
				if((nodemap[key] in itemgraphnodes) & (nodemap[it] in itemgraphnodes)):
					if(nodemap[key]>nodemap[it]):
						G.add_edge(nodemap[key],nodemap[it])
					else:
						G.add_edge(nodemap[it],nodemap[key])
					grounditemgraphnodes.add(nodemap[key])
					grounditemgraphnodes.add(nodemap[it])
		print len(grounditemgraphnodes)

	print "Number of nodes in item-item graph is ",len(itemgraphnodes), " for the category = ", category
	#print "Number of edges in item-item graph is ",len(item_item), " for the category = ",category
	print "Number of nodes in ground_item-item graph is ",len(grounditemgraphnodes), "for the category = ",category
	#print "Number of edges in ground_item-item graph is ",len(ground_item_item), " for the category = ",category

	with open('./Graphs/' + category + '/' + 'itemgraphnodes.json',"w") as f:
		json.dump(list(itemgraphnodes),f)

	with open('./Graphs/' + category + '/' + 'grounditemgraphnodes.json',"w") as f:
		json.dump(list(grounditemgraphnodes),f)
	
	with open('./Graphs/' + category + '/' + 'nodemap.json',"w") as f:
		json.dump(nodemap,f)


	a = itertools.combinations(G.nodes(),2)

	groundgraph = []

	numberofone = 0
	numberofzero = 0
	numberofneg = 0
	
	with open('./Graphs/'  + category + '/ground_item_item.csv',"a") as f:
		writer = csv.writer(f,delimiter=' ')
		for item in a:
			try:
				pathlength = nx.shortest_path_length(G,source=item[0],target=item[1])
				if(pathlength==1):
					numberofone = numberofone + 1
					writer.writerow([item[0],item[1],1])
				else:
					numberofzero = numberofzero + 1
					writer.writerow([item[0],item[1],0])
			except:
					numberofneg = numberofneg + 1
					writer.writerow([item[0],item[1],-1])

	print "ONES: ",numberofone," ZEROS: ",numberofzero," NEGATIVES: ",numberofneg
if __name__ == '__main__':
	category = sys.argv[1]
	print "category is = ",category
	cmd = os.path.join('./Graphs',category)
	if(os.path.exists(cmd)):
		shutil.rmtree(cmd)
	os.mkdir(cmd)
	getGraphs(category)
