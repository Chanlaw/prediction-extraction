import re
import numpy as np
import pandas as pd
import nltk
import tensorflow as tf
from nltk.tokenize import word_tokenize
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


data = pd.read_csv("cf_report3.txt", header=0, delimiter="\t")

def show_most_informative_features(vectorizer, clf, n=10):
	# shows the most informative features n features for a classifier. 
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)

def sentence_to_words( sentence):
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

print "Creating the bag of words..."

vectorizer = CountVectorizer(analyzer = "word",ngram_range = (1,2), max_features = 10000)

data_features = vectorizer.fit_transform(clean_sentences)
data_features = data_features.toarray()
data_features = np.concatenate((data_features, contains_numbers_digits),axis = 1)
data_features = data_features.astype(float)
target= np.array(data["label"].values)


best_dropout = 0.0
best_accuracy = 0.0

for i in [0.00, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
	test_accuracy = []
	print "-"
	print "Testing dropout = %.2f" %i
	for j in xrange(1,6):
		print "Fold %d"%j
		x_train, x_test, y_train, y_test = train_test_split(data_features,target, train_size = 0.9)
		print "Creating Neural Net..."
		classifier = tf.contrib.learn.DNNClassifier(hidden_units=[100,30], n_classes=2, dropout=i)
		print "Training Neural Net..."
		classifier.fit(x=x_train, y=y_train, steps=2000, batch_size=64)
		print "Evaluating..."
		y_pred = classifier.predict(x_test)
		test_acc = accuracy_score(y_test,y_pred)
		print "Accuracy %.05f" %test_acc
		prec, rec, fscore, support = precision_recall_fscore_support( y_test, y_pred, average='binary')
		print "Precision %.05f, Recall %.05f" %(prec, rec)
		test_accuracy.append(test_acc)
	avg_acc = (sum(test_accuracy)/len(test_accuracy))
	print "Average accuracy: %.05f" %avg_acc
	if (avg_acc > best_accuracy):
		best_dropout = i
		best_accuracy = avg_acc

print "Best dropout rate: %.2f" %best_dropout
test_accuracy = []
test_precision = []
test_recall = []

for i in xrange(1,21):
	print "-"
	print "Fold %d" % i
	x_train, x_test, y_train, y_test = train_test_split(data_features,target, train_size = 0.9)
	print "Creating Neural Net..."
	classifier = tf.contrib.learn.DNNClassifier(hidden_units=[100,30], n_classes=2, dropout=best_dropout)
	print "Training Neural Net..."
	classifier.fit(x=x_train, y=y_train, steps=2000, batch_size=64)
	print "Evaluating..."
	y_pred = classifier.predict(x_test)
	test_acc = accuracy_score(y_test,y_pred)
	print "Accuracy %.05f" %test_acc
	prec, rec, fscore, support = precision_recall_fscore_support( y_test, y_pred, average='binary')
	print "Precision %.05f, Recall %.05f" %(prec, rec)
	test_accuracy.append(test_acc)
	if (prec > 0):
		test_precision.append(prec)
	test_recall.append(rec)

print "-"
print "Number of features %d" %(data_features.shape[1])
print "Average test accuracy %.05f" %(sum(test_accuracy)/len(test_accuracy))
print "Average precision %.05f" %(sum(test_precision)/len(test_precision))
print "Average recall %.05f" %(sum(test_recall)/len(test_precision))
