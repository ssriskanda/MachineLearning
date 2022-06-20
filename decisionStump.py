#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:14:47 2020

@author: ssriskanda
"""

def decision_Stump(train, split_index):
    
    #opening data + data cleaning
    with open(train) as file:
        train = list(csv.reader(file, delimiter = '\t'))        
    for row in train:
        row.append('blank') #new column will be used to replace with classification
    train = np.array(train)
    train = np.delete(train,(0), axis = 0)
        
    #unique features and classification in data set
    feature = np.unique(train[:,split_index])
    group = np.unique(train[:,-2])
    
    #spliting data based on feature into zero and one
    zero, one = [], []
    
    #zero = np.vstack((zero, train[:,0] == feature[0]))
    #one = np.vstack((one, train[:,0] == feature[0]))
    
    for row in train:
        if row[split_index] == feature[0]:
            zero.append(row)
        elif row[split_index] == feature[1]:
            one.append(row)
    
    #left node majority vote (majority vote appended to zero)
    zero_zero = 0
    zero_one = 0
    
    for row in zero:
        if row[-2] == group[0]:
            zero_zero = zero_zero + 1
        elif row[-2] == group[1]:
            zero_one = zero_one + 1
            
    if zero_zero > zero_one:
        zero_majority = group[0]
    else:
        zero_majority = group[1]
    
    #right node majority vote (majority vote appended to one)
    one_zero = 0
    one_one = 0
    
    for row in one:
        if row[-2] == group[0]:
            one_zero = one_zero + 1
        elif row[-2] == group[1]:
            one_one = one_one + 1
    if one_zero > one_one:
        one_majority = group[0]
    else:
        one_majority = group[1]
    
    #applying majority rule to training set
    #feature zero - zero file
    #feature one -- one file  
    for row in train:
        if row[split_index] == feature[0]:
            row[-1] = zero_majority
        else:
            row[-1] = one_majority
    
    #root_node = TreeNode(None,Train)
    #left_node = Treenode(zero_majority,zero)
    #right_node = Treenode(one_majority,one)
    #root.leftnode = left
    #root.rightnode = right
            
    return train, feature, zero_majority, one_majority




#function for testing data
def testrun(test,split_index,feature,zero_majority,one_majority):
    
    #producing labels
    with open(test) as file:
        test = list(csv.reader(file, delimiter = '\t'))
        for row in test:
            row.append('blank') #new column will be used to replace with classification
        test = np.array(test)
        test = np.delete(test, (0), axis = 0)       
        
    #apply rule
    for row in test:
        if row[split_index] == feature[0]:
            row[-1] = zero_majority
        else:
            row[-1] = one_majority
    return test


#calculate error          
def errorcal(train,test):
    
    #error calculation for training dataset
    train_error = 0 
    test_error = 0
    
    #training error counting loop
    for row in train:
        if row[-2] == row[-1]:
            train_error = train_error + 0 
        else:
            train_error = train_error + 1

    #testing error counting loop
    for row in test:
        if row[-2] == row[-1]:
            test_error = test_error + 0
        else:
            test_error = test_error + 1
            
    #calculations
    train_error = (train_error)/len(train)        
    test_error = (test_error)/len(test)
    
    return train_error, test_error


def main():
        
    prediction,feature,zero_majority, one_majority = decision_Stump(train, split_index)
    actual = testrun(test,split_index,feature,zero_majority,one_majority)
    train_error, test_error = errorcal(prediction,actual)
    
    #output labels 
    f = open(train_out, "w+")
    for row in prediction:
        f.write("%s\n"%(row[-1]))
    f.close()
    
    g = open(test_out, "w+")
    for row in actual:
        g.write("%s\n"%(row[-1]))
    g.close()        
    
    h = open(metrics_out, "w+")
    h.write("error(train): %f\n"%(train_error))
    h.write("error(test): %f"%(test_error))
    h.close()        
    

if __name__ == '__main__':
    
    import sys
    import csv
    import numpy as np
    
    train = sys.argv[1]
    test = sys.argv[2]
    split_index = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]
    
    main()
    
