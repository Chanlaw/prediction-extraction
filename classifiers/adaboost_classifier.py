import re
import pandas as pd 
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Trains an AdaBoost classifier using decision trees, naive Bayes, or LogisticRegression

decisionTree = DecisionTreeClassifier(max_depth=3)
naiveBayes = MultinomialNB
logisticReg = LogisticRegression

#Set this to the type of estimator you'd like to use
weak_learner = DecisionTreeClassifier

data = pd.read_csv("cf_report.tsv", header=0, delimiter="\t", quoting=3)

def sentence_to_words( sentence ):
	# converts a raw sentence to a string of words delimited by spaces, with

	letters_only = re.sub("[^a-zA-Z\s]", "", sentence)
	words = letters_only.lower().split()
	return( " ".join( words ))

num_sentences = data["sentence"].size
clean_sentences = []

print "Cleaning sentences..."

for i in xrange( 0, num_sentences):
	if ( (i+1)%1000 == 0):
		print "Cleaning sentence %d of %d" % ( i+1, num_sentences)
	clean_sentences.append( sentence_to_words( data["sentence"][i]))

print
print "Creating the bag of words...\n"

vectorizer = CountVectorizer(analyzer = "word",ngram_range = (1,1), max_features = 1000)

data_features = vectorizer.fit_transform(clean_sentences)
data_features = data_features.toarray()
 
train_accuracy = []
test_accuracy = []
test_precision = []
test_recall = []

for i in xrange(1,21):
	#Perform 20-fold cross validation
	X_train, X_test, y_train, y_test = train_test_split(data_features, data["label"], train_size = 0.9)


	max_accuracy = 0.0
	best_C = 0.01
	print "\n-"
	print "Fold %d" % i
	print "Training AdaBoost Classifier...\n"

	clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=30), n_estimators=200)
	clf = clf.fit(X_train, y_train)
	train_acc = clf.score(X_train, y_train)
	print "Train Accuracy %.05f" % train_acc
	test_acc = clf.score(X_test, y_test)
	print "Test Accuracy %.05f" % test_acc
	prec, rec, fscore, support = precision_recall_fscore_support( y_test, clf.predict(X_test), average='binary')
	print "Precision %.05f, Recall %.05f" %(prec, rec)
	test_accuracy.append(test_acc)
	train_accuracy.append(train_acc)
	if (prec > 0):
		test_precision.append(prec)
	test_recall.append(rec)
print "-"
print "Number of features %d" %len(vectorizer.get_feature_names())
print "Average train accuracy %.05f" %(sum(train_accuracy)/len(train_accuracy))
print "Average test accuracy %.05f" %(sum(test_accuracy)/len(test_accuracy))
print "Average precision %.05f" %(sum(test_precision)/len(test_precision))
print "Average recall %.05f" %(sum(test_recall)/len(test_precision))