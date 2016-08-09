import re
import pandas as pd 
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support

# Support Vector Machine Classifier

data = pd.read_csv("cf_report3.txt", header=0, delimiter="\t", quoting=3)

def show_most_informative_features(vectorizer, clf, n=10):
	# shows the most informative features n features for a classifier. 
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)

def sentence_to_words( sentence ):
	# tokenizes sentence using the given tokenizer
	sentence=sentence.decode('utf-8')
	words = word_tokenize(sentence)
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

vectorizer = CountVectorizer(analyzer = "word", max_features = 1000)

data_features = vectorizer.fit_transform(clean_sentences)
data_features = data_features.toarray()

train_accuracy = []
test_accuracy = []
test_precision = []
test_recall = []

kernel_type = 'rbf'

for i in xrange(1,21):
	#Perform 20-fold cross validation
	X_train, X_test, y_train, y_test = train_test_split(data_features, data["label"], train_size = 0.9)
	X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size = 0.8889)

	max_accuracy = 0.0
	best_C = 0.01

	print "\n-"
	print "Fold %d" % i
	print "Training Support Vector Machine...\n"

	for c in [ 0.1, 0.3, 0.5, 1, 3, 5, 10, 30, 50, 100, 300]:
		print "Validating with C=%.2f" %c
		clf = SVC( tol=1E-6, C=c, kernel=kernel_type, n_jobs=-1)
		clf = clf.fit(X_train, y_train)
		train_acc = clf.score(X_train, y_train)
		print "Train Accuracy %.05f" % train_acc
		valid_acc = clf.score(X_valid, y_valid)
		print "Validation Accuracy %.05f" % valid_acc
		if ( valid_acc > max_accuracy ):
			best_C = c
			max_accuracy = valid_acc

	print
	print "Training final Support Vector Machine model with C=%.2f" %best_C
	clf = SVC( tol=1E-6, C=best_C, kernel = kernel_type, n_jobs=-1)
	clf = clf.fit(X_train, y_train)
	train_acc = clf.score(X_train, y_train)
	print "Train Accuracy %.05f" % train_acc
	valid_acc = clf.score(X_valid, y_valid)
	print "Valid Accuracy %.05f" % valid_acc
	test_acc = clf.score(X_test, y_test)
	print "Test Accuracy %.05f" % test_acc
	prec, rec, fscore, support = precision_recall_fscore_support( y_test, clf.predict(X_test), average='binary')
	print "Precision %.05f, Recall %.05f" %(prec, rec)
	test_accuracy.append(test_acc)
	train_accuracy.append(train_acc)
	if (prec > 0):
		test_precision.append(prec)
	test_recall.append(rec)
	if (kernel_type == 'linear'):
		print "Most informative features:"
		show_most_informative_features(vectorizer, clf)

print "-"
print "Number of features %d" %len(vectorizer.get_feature_names())
print "Average train accuracy %.05f" %(sum(train_accuracy)/len(train_accuracy))
print "Average test accuracy %.05f" %(sum(test_accuracy)/len(test_accuracy))
print "Average precision %.05f" %(sum(test_precision)/len(test_precision))
print "Average recall %.05f" %(sum(test_recall)/len(test_precision))
