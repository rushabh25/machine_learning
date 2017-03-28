#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

###
# return to_message_ratio
###
def get_to_message_ratio(from_poi_to_this_person, to_messages):
    try:
        to_message_ratio = (1.0 * int(from_poi_to_this_person) / int(to_messages))
    except:
        to_message_ratio= 0.0
    return to_message_ratio

###
# return fromo_message_ratio
###
def get_from_message_ratio(from_this_person_to_poi, from_messages):
    try:
        from_message_ratio = (1.0 * int(from_this_person_to_poi) / int(from_messages))
    except:
        from_message_ratio= 0.0
    return from_message_ratio

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'from_message_ratio', 'to_message_ratio', 'bonus', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

###
# logic to remove the outlier
###
##salary_list = []
##for key,value in data_dict.iteritems():
##    if(value["salary"] != 'NaN'):
##        salary_list.append(value["salary"])
##print 'max and min of salaries: %d and %d' % (max(salary_list), min(salary_list))
##max_salary = max(salary_list)
##max_salary_key = ''
##
##for key, value in data_dict.iteritems():
##    if(value['salary'] == max_salary):
##        print 'key with maximum salary of %d is %s' % (max_salary, key)
##        max_salary_key = key
##        continue
#print len(data_dict)
data_dict.pop('TOTAL')
#print len(data_dict)

for key,value in data_dict.iteritems():
    data_dict[key]["to_message_ratio"]=get_to_message_ratio(data_dict[key]['from_poi_to_this_person'], data_dict[key]['to_messages'])
    data_dict[key]["from_message_ratio"]=get_from_message_ratio(data_dict[key]['from_this_person_to_poi'], data_dict[key]['from_messages'])

print data_dict.iteritems().next()    

  
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
temp = 0
    
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list,remove_NaN=True, remove_all_zeroes=True)
labels, features = targetFeatureSplit(data)


###
# Plot data
# poi as red and non poi as blue
# clearly there is an outlier, lets catch it and remove it from the dictionary
###
colors = ['b', 'r']

for i in range(0, len(labels)):
    plt.scatter( features[i][0], features[i][1], color=colors[int(labels[i])] )
plt.xlabel(features_list[1])
plt.ylabel(features_list[2])           
#plt.show()



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
##from sklearn.naive_bayes import GaussianNB
##clf = GaussianNB()



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


for i in range(0, len(labels_train)):
    plt.scatter( features_train[i][0], features_train[i][1], color=colors[int(labels_train[i])] )
plt.xlabel(features_list[1])
plt.ylabel(features_list[2])           
#plt.show()

##clf.fit(features_train, labels_train)
##pred = clf.predict(features_test)
##print 'accuracy_score with Gaussian Naive Bayes is: %f' % accuracy_score(labels_test, pred)

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

###
#
# Decision Tree Classifier
#
###
from sklearn import tree
from sklearn.feature_selection import SelectKBest
classifier = tree.DecisionTreeClassifier(criterion='gini')

steps = [('decision_tree', classifier)]
pipeline = Pipeline(steps)

param_grid = dict(
    max_depth=np.arange(2,20),
    min_samples_split = np.arange(2,15)
    )

grid_search = GridSearchCV(classifier, param_grid, scoring='recall')
grid_search.fit(features_train, labels_train)
pred2 = grid_search.predict(features_test)

print 'Best score: %0.3f' % grid_search.best_score_
print 'Best parameters set:'
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print '\t%s: %r' % (param_name, best_parameters[param_name])

#print features_train[0:5], features_test[0:5]
clf2 = tree.DecisionTreeClassifier(min_samples_split=2, max_depth=7, criterion='gini')
clf2 = clf2.fit(features_train, labels_train)
pred = clf2.predict(features_test)
###
#
#   Random Forest Ensemble Method
#
###

##from sklearn.ensemble import RandomForestClassifier
##from sklearn.pipeline import Pipeline
##from sklearn.feature_selection import SelectKBest
##
##select = SelectKBest()
##classifier = RandomForestClassifier()
##
##steps = [('feature_selection', select), ('random_forest', classifier)]
##
##pipeline = Pipeline(steps)
##
##param_grid = dict(
##    feature_selection__k=[2,3,4,5],
##    random_forest__n_estimators=np.arange(10,30),
##    random_forest__min_samples_split = np.arange(2,15)
##    )
##
##grid_search = GridSearchCV(pipeline, param_grid, scoring='recall')
##grid_search.fit(features_train, labels_train)
##pred2 = grid_search.predict(features_test)
##
##print 'Best score: %0.3f' % grid_search.best_score_
##print 'Best parameters set:'
##best_parameters = grid_search.best_estimator_.get_params()
##for param_name in sorted(param_grid.keys()):
##    print '\t%s: %r' % (param_name, best_parameters[param_name])

### As a result of GridSearch here are the best parameters for RandomForets
#   Best score: 0.862
#   Best parameters set:
#   max_depth: 5
#   min_samples_split: 5
#   n_estimators: 6
###


from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
print classification_report(labels_test, pred)

# function for calculation ratio of true positives
# out of true positives and false negatives
#print 'accuracy_score with Random Forest Tree Classifier is: %f' % accuracy_score(labels_test, pred2)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf2, my_dataset, features_list)
