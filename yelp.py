

import json
import nltk
import numpy as np
from nltk import pos_tag, word_tokenize
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from nltk.corpus import stopwords

def tokenizeReview(reviewList):
	tokenizedWords={}
	for review in reviewList:

		tokenizedWords[review[0]]=word_tokenize(review[1])
		#print tokenizedWords
	return tokenizedWords 

def buildLexicon(tokenizedWords):
	lexicon=set()
	i=1
	for i in range(1,len(tokenizedWords)+1):
		lexicon.update(tokenizedWords[i])
	return lexicon

def tf(word,tokenizedWords):
	return tokenizedWords.count(word)

def createTfIdfMatrix(tokenizedWords):
	lexicon=buildLexicon(tokenizedWords)
	tf_vector={}
	for i in range(1, len(tokenizedWords)+1):
		tf_vector[i]=[tf(word,tokenizedWords[i]) for word in lexicon]
	return lexicon,tf_vector

def createTags(dictSent):
	tags=dictSent.values()
	return tags

def classification(trainVecs,trainTags):
	clf = OneVsRestClassifier(SVC(C=1, kernel = 'linear', gamma=1, verbose= False, probability=False))
	clf.fit(trainVecs, trainTags)
	print "Classifier Trained..."
	predicted = cross_validation.cross_val_predict(clf, trainVecs, trainTags, cv=5)
	print "Cross Fold Validation Done..."
	print "accuracy score: ", metrics.accuracy_score(trainTags, predicted)
	print "precision score: ", metrics.precision_score(trainTags, predicted,pos_label=None,average='weighted')
	print "recall score: ", metrics.recall_score(trainTags, predicted,pos_label=None,average='weighted')
	print "classification_report: \n ", metrics.classification_report(trainTags, predicted)
	print "confusion_matrix:\n ", metrics.confusion_matrix(trainTags, predicted)
	return


def removeStopwords(tokenizedWords):
	for i in range(1,len(tokenizedWords)+1):
		filteredWords=[word for word in tokenizedWords[i] if word not in stopwords.words('english')]
		tokenizedWords[i]=filteredWords
	return tokenizedWords

def main():
	f =open("C:\Users\samee\Desktop\ILS z 604\Yelp data challenge\yelp_academic_dataset_review.json")

	line=f.readline()
	dictSent={}
	dictStar={}
	reviewList=[]
	i=1
	sentiment='negative'
	while line:
		line=f.readline()
		review=json.loads(line)
		index=i
		star=review["stars"]
		text= review["text"]
		if star>3:
			sentiment='1.0' #positive
		else:
			sentiment='0.0'  #negative
		dictSent[index]=sentiment
		dictStar[index]=star
		reviewList.append([index,text])
		i+=1
		if i==5001:
			break	
	f.close()
	print "Dataset Loaded..."
	tokenizedWords=tokenizeReview(reviewList)
	print "Reviews Tokenized..."
	print"\n Classification without any processing"
	print "#"*70
	lexicon,tfVector=createTfIdfMatrix(tokenizedWords)
	print "TF Matrix Created..."
	print "length of vector : ",len(tfVector[1])
	tags=createTags(dictSent)
	trainVecs=np.array(tfVector.values())
	trainTags=np.array(tags)
	classification(trainVecs,trainTags)
	print"#"*70
	print"\n Classification after removing stop words"
	print "#"*70
	tokenizedWords=removeStopwords(tokenizedWords)
	lexicon,tfVector=createTfIdfMatrix(tokenizedWords)
	print "TF Matrix Created..."
	print "length of vector : ",len(tfVector[1])
	tags=createTags(dictSent)
	trainVecs=np.array(tfVector.values())
	trainTags=np.array(tags)
	classification(trainVecs,trainTags)
	print"#"*70
	print "Classification into 5 Classes"
	print"#"*70
	tags=createTags(dictStar)
	trainTags=np.array(tags)
	classification(trainVecs,trainTags)

	# #reviewPosTag=tokenizeReview(reviewList)

if __name__=="__main__":
	main()