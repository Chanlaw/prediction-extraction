import nltk
import sys
reload(sys)
sys.setdefaultencoding("iso-8859-1")

print "sentence"
segmenter = nltk.data.load('tokenizers/punkt/english.pickle')

for text in sys.stdin :
	if len(text) >1:
		print ('\n'.join(segmenter.tokenize(text.strip())))