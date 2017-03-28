#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary','bonus','total_stock_value','exercised_stock_options'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

###
# logic to remove the outlier
###
salary_list = []
for key,value in data_dict.iteritems():
    if(value["salary"] != 'NaN'):
        salary_list.append(value["salary"])
print 'max and min of salaries: %d and %d' % (max(salary_list), min(salary_list))
max_salary = max(salary_list)
max_salary_key = ''

for key, value in data_dict.iteritems():
    if(value['salary'] == max_salary):
        print 'key with maximum salary of %d is %s' % (max_salary, key)
        max_salary_key = key
        continue
#print len(data_dict)
data_dict.pop(max_salary_key)
#print len(data_dict)

  
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
print my_dataset.iteritems().next()
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list sort_keys = True)
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
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()



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

clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print 'accuracy_score with Gaussian Naive Bayes is: %f' % accuracy_score(labels_test, pred)

from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

###
#
# Decision Tree Classifier
#
###
from sklearn import tree
from sklearn.feature_selection import SelectKBest
classifier = tree.DecisionTreeClassifier(criterion='entropy')

steps = [('decision_tree', classifier)]
pipeline = Pipeline(steps)

param_grid = dict(
    decision_tree__max_depth=np.arange(2,20),
    decision_tree__min_samples_split = np.arange(2,15)
    )

grid_search = GridSearchCV(pipeline, param_grid, scoring='recall')
grid_search.fit(features_train, labels_train)
pred2 = grid_search.predict(features_test)

print 'Best score: %0.3f' % grid_search.best_score_
print 'Best parameters set:'
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
    print '\t%s: %r' % (param_name, best_parameters[param_name])

clf2 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=5)

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


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
print classification_report(labels_test, pred2)
#print 'accuracy_score with Random Forest Tree Classifier is: %f' % accuracy_score(labels_test, pred2)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf2, my_dataset, features_list)
