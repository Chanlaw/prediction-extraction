import nltk
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

print "sentence"

for text in sys.stdin :
	segmenter = nltk.data.load('tokenizers/punkt/english.pickle')
	print ('\n'.join(segmenter.tokenize(text.strip())))