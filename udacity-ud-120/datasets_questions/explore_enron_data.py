#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import math

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
enron_data_filtered = {}

print len(enron_data)
print enron_data.items()[0]
print len(enron_data.values()[0].keys())

print enron_data.values()[0]['poi']

#for key, val in enron_data.iteritems():
#    if(val['poi'] == 0):
#        enron_data_filtered[key] = val


#print len(enron_data_filtered)

print enron_data["PRENTICE JAMES"]["total_stock_value"]
print enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

print enron_data["LAY KENNETH L"]["total_payments"]
print enron_data["SKILLING JEFFREY K"]["total_payments"]
print enron_data["FASTOW ANDREW S"]["total_payments"]

print type(enron_data.values()[0]["salary"])
for key, val in enron_data.iteritems():
    if ("@" not in val['email_address']):
        enron_data_filtered[key] = val

print len(enron_data_filtered)
