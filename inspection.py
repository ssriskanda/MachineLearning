#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 23:21:36 2020

@author: ssriskanda
"""
def open_file(train):
    #train: training set of the data

    with open(train) as file:
        train = list(csv.reader(file, delimiter = '\t'))        
    for row in train:
        row.append("") #new column will be used to replace with classification
    train = np.array(np.delete(train,(0), axis = 0))#drop name of headers
        
    return train

def entropy(data,index):
    #calculates entrophy of specific column
    feature = []
    y = np.array(np.unique(data[:,int(index)])) #obtaining unique features of attribute
    for item in y:
        feature.append(item)
    #counting how many of each appear
    count_zero = 0
    count_one = 0
    
    for row in data:
        if row[index] == feature[0]:
            count_zero += 1
        else:
            count_one += 1
    count = [count_zero, count_one]
    #entropy = -np.sum((count[i]/len(data))*np.log2(count[i]/len(data)) for i in range(len(feature)))
    entropy = -sum((count[i]/len(data))*np.log2(count[i]/len(data)) for i in range(len(feature)))

    return entropy



def majority(data):
    
    feature, count = np.unique(data[:,-2], return_counts = True)
    
    if count[0] > count[1]:
        majority = feature[0]
    elif count[0] == count[1]:
        majority = feature[1]
    elif count[0] < count[1]:
        majority = feature[1]
    
    return majority

def application(data,majority):
    
    for row in data:
        row[-1] = majority
    
    return data
        
def errorcal(data):
    
    #error calculation for training dataset
    train_error = 0 
    
    #training error counting loop
    for row in data:
        if row[-2] == row[-1]:
            train_error = train_error + 0 
        else:
            train_error = train_error + 1
            
    #calculations
    train_error = (train_error)/len(data)        
    
    return train_error   
    
    
def main():
    
    data = open_file(train)
    
    entrop = entropy(data,-2)
    
    vote = majority(data)
    
    data1 = application(data, vote)
    
    train_error = errorcal(data1)
        
    #output labels 
    h = open(metrics_out, "w+")
    h.write("entropy: %f\n"%(entrop))
    h.write("error: %f"%(train_error))
    h.close()   
    
       

if __name__ == '__main__':
    
    import sys
    import csv
    import numpy as np
    
    train = sys.argv[1]
    metrics_out = sys.argv[2]
    
    main()

