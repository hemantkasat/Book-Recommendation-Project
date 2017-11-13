import json
import sys
import os

f = open('../amazon-meta.txt')

data = f.read()

pitems = data.split('\r\n\r\n')


with open('./ProductDetails/ProductDetails.json',"r") as f:
	ProductDetail = json.load(f)



def getBooksbycategory(category):
	Books = []
	print "Category is ", category
	for item in pitems:
		if(len(item.split('\r\n'))>4):
			if "group: Book" == item.split('\r\n')[3].strip():
				if('NoReviews' in category):
					asin  = item.split('\r\n')[1].strip().split(':')[1].strip()
					try:
						det = ProductDetail[asin]
						for it in item.split('\r\n'):
							if 'reviews:' in it:
								noreviews = int(it.strip().split('reviews:')[1].strip().split('total:')[1].strip().split('downloaded:')[0])
								if(noreviews>=int(category[9:])):
									Books.append(item)
					except:
						pass
				elif (category == 'All'):
					asin  = item.split('\r\n')[1].strip().split(':')[1].strip()
					try:
						det = ProductDetail[asin]
						Books.append(item)
					except:
						pass


				else:
					asin  = item.split('\r\n')[1].strip().split(':')[1].strip()
					if category in item:
						try:
							det = ProductDetail[asin]
							Books.append(item)
						except:
							pass

	return Books




if __name__ == "__main__":
	category = sys.argv[1]
	books = getBooksbycategory(category)
	print "Number of Items in this category are ", len(books)
	with open('metadatabycategory/' + category + '.json',"w") as f:
		json.dump(books,f)
