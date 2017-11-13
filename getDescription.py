import json


def getAsins():

	Description = {}
	f = open('../amazon-meta.txt')
	data = f.read()
	pitems = data.split('\r\n\r\n')
	for item in pitems:
		if "ASIN" in item:
			asin = item.split('\r\n')[1].strip().split('ASIN:')[1].strip()
			if asin not in Description.keys():
				Description[asin] = []
		print len(Description.keys())
	return Description

def getDesciption(Description):

	f = open('../output.strict',"r")
	for line in f:
		product = json.loads(line)
		if('description' in product.keys()):

			if product['asin'] in Description.keys():
				print product
				Description[product['asin']] = product
	return Description


if __name__ == '__main__':
	Description = getAsins()
	Description = getDesciption(Description)
	print len(Description.keys())