#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 20:00:08 2020

@author: ssriskanda
"""

#create dictionary

def create_dictionary(index_to_tag, index_to_word):
    #creating dictionary for each text file
    dictionary_1 = dict()
    dictionary_2 = dict()
    
    #loop through for dictionary
    for i in range(len(index_to_word)):
        dictionary_1[index_to_word[i]] = i
    for i in range(len(index_to_tag)):
        dictionary_2[index_to_tag[i]] = i
    
    #also getting the length of each dictionary for later
    tag_length = len(index_to_tag)
    word_length = len(index_to_word)
        
    return dictionary_1, dictionary_2, tag_length, word_length
    
    

#prior probability
def prior(train, dict_2):    
    #collecting list of first words in data set
    firsts = []
    for row in train:
        firsts.append(row[0])
    
    #obtaining tag of first words in dataset           
    tag = []
    for thing in firsts:
        lines = thing.split("_")
        tag.append(lines[1])
    tag = np.array(tag)
    
    #creating a counter to see how many times each tag apperas
    counter = np.zeros(len(dict_2))
    for i in range(len(tag)):
        counter[dict_2[tag[i]]] += 1
    
    #adding pseudo count
    counter = counter + 1
    
    #dividing each one over the total
    denom = sum(counter)
    prior = counter/denom

    return prior


def emission(train, dict_1, dict_2, tag_length, word_length):
    #creating empty matrix
    B = np.zeros((tag_length, word_length))
    
    #doing a count for each tag
    for i in range(len(train)):
        for j in range(len(train[i])):
            line = train[i][j].split("_")
            B[dict_2[line[1]]][dict_1[line[0]]] += 1
            
    #adding psuedo count
    B = B + 1
    
    #diving each over the total
    denom = B.sum(axis = 1)[:,None]
    B = B/denom
    return B



    
def transition(train, dict_1, dict_2, tag_length):
    #empty transition 
    A = np.zeros((tag_length, tag_length))
    
    for i in range(len(train)):
        for j in range(len(train[i])):
            word,tag = train[i][j].split("_")
            if j != len(train[i]) -1 :
                word_2, tag_2 = train[i][j+1].split("_")
                A[dict_2[tag]][dict_2[tag_2]] += 1
   
    #adding psuedo count
    A = A+1
    #diving each over the total
    denom = A.sum(axis = 1)[:,None]
    A = A/denom
    
    return A
    
    
    

def main(): 
    

    import numpy as np
    import csv

    #File
    #predicted_import = open("full_predicted.txt", "r").read().splitlines()
    #B = np.array(list(csv.reader(open("full_hmmemit.txt", "r"), delimiter = " "))).astype(float)
    #pi = np.array(list(csv.reader(open("full_hmmprior.txt", "r"), delimiter = " "))).astype(float)
    #A = np.array(list(csv.reader(open("full_hmmtrans.txt", "r"), delimiter = " "))).astype(float)
    
    train_import = open(trai, "r").read().splitlines()
    index_to_word = np.array(open(word).read().splitlines())
    index_to_tag = np.array(open(tag).read().splitlines())
    #validation_import = open("full_validation.txt", "r").read().splitlines()
    #train_import = open("full_train.txt", "r").read().splitlines()

    dict_1, dict_2, tag_length, word_length = create_dictionary(index_to_tag, index_to_word)    

    train = []
    for row in train_import:
        line = row.split(" ")
        train.append(line)
    

    pi = prior(train, dict_2)
    B = emission(train, dict_1, dict_2, tag_length, word_length)
    A = transition(train, dict_1, dict_2, tag_length)
    
    
    f = open(pri, "w+")
    for row in pi:
        f.write("%f\n"%(row))
        
    g = np.savetxt(emit, B, fmt="%f") 
    h = np.savetxt(trans, A, fmt="%f") 
    


if __name__ == '__main__':
    
    ##import appropiate packages
    import sys
    import csv
    import numpy as np
    
    
    trai = sys.argv[1]
    word = sys.argv[2]
    tag = sys.argv[3]
    pri = sys.argv[4]
    emit = sys.argv[5]
    trans = sys.argv[6]
    
    
    main()





