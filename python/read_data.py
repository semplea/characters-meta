# coding: utf-8
import re
import unicodedata
import codecs

def removeAccents(in_str):
	encoding = "utf-8"
	u_str = in_str.decode(encoding)
	temp = unicodedata.normalize('NFKD', u_str)
	norm_str = temp.encode('ASCII', 'ignore')
	return norm_str


def readData():
	"""Read data and create sublists everytime a line starts with #
	Return dict with category -> list(words).
	Keys of dictionary returned are 'tromper', 'nutrition', 'dormir', 'raison', 'tuer', 'metiers', 'vouloir', 'pensee',
	'emotions', 'guerir', 'relations', 'soupir', 'etats', 'parole', 'salutations', 'foi'
	"""
	fname = "/home/alexis/Documents/EPFL/MS3/Project/python/classifiersdata/proximitywordclasses/character.txt"

	with codecs.open(fname, 'r', 'utf8') as f:
		content = []
		for line in f:
			content.append(line[:-1]) # get rid of newlines

	# Split into sublists with header line starting with # (hash-symbol)
	acc = content[0]
	categories = [[acc]]
	for line in content[1:]:
		if line.startswith('#'):
			acc = line
			categories.append([acc])
		else:
			categories[-1].append(line)

	# Construct dict with key from header of sublists
	cats_dict = dict()
	rx = re.compile("[\w]+", re.UNICODE)
	for lst in categories[1:]:
		str_name = lst[0]
		#str_name = removeAccents(str_name)
		cat_name = rx.findall(str_name)[0]
		cats_dict[cat_name] = lst[1:]
	return cats_dict
