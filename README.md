# Yelp-Dataset-Classification-using-SVM
Classification of Yelp Reviews into Stars they can get

Approach followed by me:
1.	Read the dataset and separate the reviews into positive and negative on the basis of stars
2.	If Stars>3 then 1.0 else 0.0
3.	Create Feature vector
4.	Build the classifier
5.	Do 5 fold cross validation
6.	Try some trial and error using pos tagging and rejecting stop words for step 4 and 5

I also implemented the 5 class classification by using SVM Classifier, the approach was similar 
Below is the Output

Output:
Dataset Loaded...
Reviews Tokenized...
 Classification without any processing
######################################################################
TF Matrix Created...
length of vector :  26051
Classifier Trained...
Cross Fold Validation Done...
accuracy score:  0.7674
precision score:  0.76834786108
recall score:  0.7674
classification_report:
               precision    recall  f1-score   support

        0.0       0.67      0.68      0.68      1777
        1.0       0.82      0.81      0.82      3223
avg / total       0.77      0.77      0.77      5000

confusion_matrix:
  [[1211  566]
 [ 597 2626]]
######################################################################
 Classification after removing stop words
######################################################################
TF Matrix Created...
length of vector :  25924
Classifier Trained...
Cross Fold Validation Done...
accuracy score:  0.7632
precision score:  0.764003111764
recall score:  0.7632
classification_report:
               precision    recall  f1-score   support

        0.0       0.66      0.67      0.67      1777
        1.0       0.82      0.81      0.82      3223

avg / total       0.76      0.76      0.76      5000

confusion_matrix:
  [[1198  579]
 [ 605 2618]]
######################################################################
Classification into 5 Classes
######################################################################
Classifier Trained...
Cross Fold Validation Done...
accuracy score:  0.4504
precision score:  0.445510219647
recall score:  0.4504
classification_report:
               precision    recall  f1-score   support

          1       0.51      0.49      0.50       541
          2       0.29      0.24      0.26       497
          3       0.29      0.27      0.28       739
          4       0.38      0.40      0.39      1388
          5       0.58      0.61      0.59      1835

avg / total       0.45      0.45      0.45      5000

confusion_matrix:
  [[ 265   87   48   65   76]
 [ 100  121  102   99   75]
 [  57  101  202  248  131]
 [  45   71  201  549  522]
 [  48   44  137  491 1115]]

