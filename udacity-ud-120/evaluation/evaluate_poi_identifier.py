#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 


### split features/labels into training and test set
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size = 0.3, random_state = 42)

### Create a decision tree classifier
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
#print clf.score(features, labels)
filtered_list = list(filter(lambda x: x == 1.0, pred))
print len(filtered_list), len(features_test)
### Quiz: Accuracy of biased identifier
from sklearn.metrics import accuracy_score
zeros_list = [0] * len(pred)
print accuracy_score(labels_test, zeros_list)
###
### metrics

print accuracy_score(labels_test, pred)
counter = 0
for i in range(0, len(labels_test)):
    if(labels_test[i] == 1 and pred[i] == 1):
        counter = counter + 1

print counter

from sklearn.metrics import precision_score, recall_score
print precision_score(labels_test, pred)
print recall_score(labels_test, pred)
