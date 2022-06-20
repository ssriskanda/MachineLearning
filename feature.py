#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 17:46:26 2020

@author: ssriskanda
"""


def openfile(train, test, valid):
    
    with open(train) as file:
        train = list(csv.reader(file, delimiter = '\t'))


    with open(test) as file:
        test = list(csv.reader(file, delimiter = '\t'))
  

    with open(valid) as file:
        valid = list(csv.reader(file, delimiter = '\t'))
  
    
    return train, test, valid
    

def open_dict(dictionary):
    with open(dictionary, "r") as file:
        temp = list(csv.reader(file, delimiter = '\n'))
        
    return temp
    

def create_dict(temp):
    
    dic = []
    dictionary = {}

    for i in range(len(temp)):
        for item in temp[i]:
            item = item.split(" ")
            dic.append(item)

    for i in range(len(dic)):
            dictionary[dic[i][0]] = dic[i][1]
    
    keys = dictionary.keys()
    
    return dictionary,keys

def model_one(data, dictionary, keys):
    
    final = []
    
    for i in range(len(data)):
        temp_row = []
        temp_row.append(data[i][0])
        review = data[i][1].split(" ")
        
        for word in review:
            if word in keys and ("%s:%d"%(dictionary[word], 1)) not in temp_row:
                temp_row.append(("%s:%d"%(dictionary[word], 1)))
        final.append(temp_row)
        
        
    return final
    

def model_two(data, dictionary, keys):
        
    final = []
    
    for i in range(len(data)):
        temp_row = []
        temp_row.append(data[i][0])
        review = data[i][1].split(" ")
    
        for word in review:
            num = review.count(word)
            if word in keys and ("%s:%d"%(dictionary[word], 1)) not in temp_row  and num < 4:
                temp_row.append(("%s:%d"%(dictionary[word], 1)))
    
        final.append(temp_row)
        
    return final
        
        
       
def main(): 

    #import csv
    #import numpy as np

    #train = "/Users/ssriskanda/Documents/Carnegie_Mellon/Second_Year/First_Semester/10601/hw4/handout/smalldata/train_data.tsv"
    #test = "/Users/ssriskanda/Documents/Carnegie_Mellon/Second_Year/First_Semester/10601/hw4/handout/smalldata/test_data.tsv"
    #valid = "/Users/ssriskanda/Documents/Carnegie_Mellon/Second_Year/First_Semester/10601/hw4/handout/smalldata/valid_data.tsv"


    #with open("dict2.txt", "r") as file:
    #    temp = list(csv.reader(file, delimiter = '\n'))
    
    train_dat, test_dat, valid_dat = openfile(trai,tes,vali)
    
    temp_dict = open_dict(diction)
    
    dictionary, keys = create_dict(temp_dict)
    
    if feature == 1:
       train = model_one(train_dat, dictionary, keys)
       test = model_one(test_dat, dictionary, keys)
       valid = model_one(valid_dat, dictionary, keys)
    elif feature == 2: 
       train = model_two(train_dat, dictionary, keys)
       test = model_two(test_dat, dictionary, keys)
       valid = model_two(valid_dat, dictionary, keys)        

    
    #output for training

    with open(train_out, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for item in train:
            tsv_writer.writerow(item)  
    
    with open(valid_out, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for item in valid:
            tsv_writer.writerow(item)  
            
    with open(test_out, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for item in test:
            tsv_writer.writerow(item)      

        
if __name__ == '__main__':
    
    ##import appropiate packages
    import sys
    import csv
    
    trai = sys.argv[1]
    vali = sys.argv[2]
    tes = sys.argv[3]
    diction = sys.argv[4]
    train_out = sys.argv[5]
    valid_out = sys.argv[6]
    test_out = sys.argv[7]
    feature = int(sys.argv[8])
    

    
    main()
    

    
       
        
        

    
    

