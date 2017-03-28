#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    #print predictions, ages, net_worths
    cleaned_data = []


    from collections import namedtuple
    from operator import attrgetter
    Outlier = namedtuple("outlier", "predictions ages net_worths error")
    seq = []
    for i in range(0,90):
        o = Outlier(predictions=predictions[i][0], ages=ages[i][0], net_worths=net_worths[i][0], error=((predictions[i][0] - net_worths[i][0])**2))
        seq.append(o)
    #print predictions[0]
    #print predictions[90]
    sorted_list = []
    sorted_list = sorted(seq, key = lambda x: int(x[3]))
    sorted_list = sorted_list[0:81]
    
    for i in range(0, len(sorted_list)):
        myTuple = (sorted_list[i].ages, sorted_list[i].net_worths, sorted_list[i].error)
        
        cleaned_data.append(myTuple)
    
    return cleaned_data

