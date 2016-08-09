import re
import pandas as pd 
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support


# Logistic regression classifier

data = pd.read_csv("cf_report3.txt", header=0, delimiter="\t")

def show_most_informative_features(vectorizer, clf, n=10):
	# shows the most informative features n features for a classifier. 
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)

def sentence_to_words( sentence ):
	# tokenizes sentence using the given tokenizer
	
	contains_numbers_digit = 1
	try:
		sentence=sentence.decode('iso-8859-1')
		if not any(i.isdigit() for i in sentence):
			contains_numbers_digit = 0
	except AttributeError:
		print sentence
	words = word_tokenize(sentence)
	return( " ".join( words ), contains_numbers_digit)

num_sentences = data["sentence"].size
clean_sentences = []
contains_numbers_digits = np.zeros((num_sentences, 1), dtype = np.int)

print "Cleaning sentences..."


for i in xrange( 0, num_sentences):
	if ( (i+1)%1000 == 0):
		print "Cleaning sentence %d of %d" % ( i+1, num_sentences)
	(clean_sentence, contains_numbers_digit) = sentence_to_words( data["sentence"][i])
	clean_sentences.append( clean_sentence)
	contains_numbers_digits[i][0] = contains_numbers_digit

print "Creating the bag of words...\n"

vectorizer = CountVectorizer(analyzer = "word",ngram_range = (1,2), max_features = 10000)

data_features = vectorizer.fit_transform(clean_sentences)
data_features = data_features.toarray()
data_features = np.concatenate((data_features, contains_numbers_digits),axis = 1)

train_accuracy = []
test_accuracy = []
test_precision = []
test_recall = []

best_classifier = LogisticRegression( tol=1E-6)
max_fscore = 0.0 

for i in xrange(1,21):
	#Perform 20-fold cross validation
	X_train, X_test, y_train, y_test = train_test_split(data_features, data["label"], train_size = 0.9)
	X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size = 0.8889)

	print X_train.shape
	max_accuracy = 0.0
	best_C = 0.0
	print "-"
	print "Fold %d" % i
	print "Training Logistic Regression Classifier...\n"

	for c in [ 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1, 3, 5, 10, 30, 50, 100]:
		print "Validating with C=%.2f" % c
		logistic = LogisticRegression( tol=1E-6, C=c)
		logistic = logistic.fit(X_train, y_train)
		train_acc = logistic.score(X_train, y_train)
		print "Train Accuracy %.05f" % train_acc
		valid_acc = logistic.score(X_valid, y_valid)
		print "Validation Accuracy %.05f" % valid_acc
		if ( valid_acc > max_accuracy ):
			best_C = c
			max_accuracy = valid_acc

	print
	print "Training final logistic regression model with C=%.2f" %best_C
	logistic = LogisticRegression( tol=1E-6, C=best_C)
	logistic = logistic.fit(X_train, y_train)
	train_acc = logistic.score(X_train, y_train)
	print "Train Accuracy %.05f" % train_acc
	valid_acc = logistic.score(X_valid, y_valid)
	print "Valid Accuracy %.05f" % valid_acc
	test_acc = logistic.score(X_test, y_test)
	print "Test Accuracy %.05f" % test_acc
	prec, rec, fscore, support = precision_recall_fscore_support( y_test, logistic.predict(X_test), average='binary')
	print "Precision %.05f, Recall %.05f" %(prec, rec)
	test_accuracy.append(test_acc)
	train_accuracy.append(train_acc)
	if (prec > 0):
		test_precision.append(prec)
	test_recall.append(rec)
	print "Most informative features:"
	show_most_informative_features(vectorizer, logistic)
	
	
print "-"
print "Number of features %d" %(data_features.shape[1])
print "Average train accuracy %.05f" %(sum(train_accuracy)/len(train_accuracy))
print "Average test accuracy %.05f" %(sum(test_accuracy)/len(test_accuracy))
print "Average precision %.05f" %(sum(test_precision)/len(test_precision))
print "Average recall %.05f" %(sum(test_recall)/len(test_precision))

# print "Most informative features:"
# show_most_informative_features(vectorizer, best_classifier)

# print 
# test = pd.read_csv("test_sentences.csv", header=0, delimiter=",")
# num_test_sentences = test["sentence"].size
# print "Cleaning sentences..."

# contains_numbers_digits = np.zeros((num_test_sentences, 1), dtype = np.int)
# clean_sentences = []
# for i in xrange( 0, num_test_sentences):
# 	(clean_sentence, contains_numbers_digit) = sentence_to_words(test["sentence"][i])
# 	clean_sentences.append(clean_sentence)
# 	contains_numbers_digits[i][0] = contains_numbers_digit

# test_features = vectorizer.transform(clean_sentences).toarray()
# test_features = np.concatenate((test_features, contains_numbers_digits), axis = 1)
# test_labels = best_classifier.predict(test_features)
# print "Labelled %d predictions in %d sentences" % (sum(test_labels), num_test_sentences)

# print "Accuracy %.05f" %(best_classifier.score(test_features, test["my_label"]))
# prec, rec, fscore, support = precision_recall_fscore_support( test["my_label"], test_labels, average='binary')
# print "Precision %.05f, Recall %.05f" %(prec, rec)
# print "Exporting to file..." 
# output = pd.DataFrame(data={"machine_label": test_labels, "my_label": test["my_label"], "sentence":test["sentence"]})
# output.to_csv( "Machine Labelled Sentences.csv", index=False)

