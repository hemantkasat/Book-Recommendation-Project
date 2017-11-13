from collections import defaultdict
import json
import sys

def getReviewsbycategory(category):
	with open('./Graphs/' + category + '/nodemap.json',"r") as f:
		nodemap = json.load(f)

	nodereviewmap = defaultdict(set)

	with open('../Books.txt') as f:
		for line in f:
			if('product/productId:' in line):
				item = line.split('product/productId:')[1].strip()

			if str(item) in nodemap.keys():
				print item
				if 'review/summary:' in line:
					summary = line.split('review/summary:')[1].strip()
				if 'review/text:' in line:
					text = line.split('review/text:')[1].strip()
					nodereviewmap[item].add(summary + '\n' + text)
	
	print "Total number of Books for which Reviews has to be found is ",len(nodemap.keys())
	print "Found Reviews for ",len(nodereviewmap.keys()), " number of books"
	with open('./Reviews/' + category + '_reviews.json',"w") as f:
		json.dump(nodereviewmap,f)

if __name__ == '__main__':
	category = sys.argv[1]	
	getReviewsbycategory(category)
