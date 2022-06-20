#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 00:45:44 2020

@author: ssriskanda
"""

import numpy as np
import csv

'''
    #File
predicted_import = open("toy_predicted.txt", "r").read().splitlines()
B = np.array(list(csv.reader(open("toy_hmmemit.txt", "r"), delimiter = " "))).astype(float)
pi = np.array(list(csv.reader(open("toy_hmmprior.txt", "r"), delimiter = " "))).astype(float)
A = np.array(list(csv.reader(open("toy_hmmtrans.txt", "r"), delimiter = " "))).astype(float)
    
train_import = open("toy_train.txt", "r").read().splitlines()
index_to_word = np.array(open("toy_index_to_word.txt").read().splitlines())
index_to_tag = np.array(open("toy_index_to_tag.txt").read().splitlines())
validation_import = open("toy_validation.txt", "r").read().splitlines()
'''

def create_dictionary(index_to_tag, index_to_word):
    dict_1 = dict()
    dict_2 = dict()
    for i in range(len(index_to_word)):
        dict_1[index_to_word[i]] = i
    for i in range(len(index_to_tag)):
        dict_2[index_to_tag[i]] = i
    
    tag_length = len(index_to_tag)
    word_length = len(index_to_word)
        
    return dict_1, dict_2, tag_length, word_length

def getList(dict_2): 
    keys = [] 
    for key in dict_2.keys():
        keys.append(key)
    return keys


def valid(validation):
    
    true_labels = []
    words = []
    
    for item in validation:
        line = item.split("_")
        words.append(line[0])
        true_labels.append(line[1])
    
    return true_labels, words                                                 
                                                            
#find alpha
def create_alpha(i, end, alpha, pi, A, B, dict_1, dict_2, words):
    if i == end:
        return alpha
    if i == 0:
        index = B[:,dict_1[words[i]]]
        pi = pi.reshape((pi.shape[0], ))
        alpha_part = np.multiply(index, pi)
        alpha[i] = alpha_part
        create_alpha(i+1, len(words), alpha, pi, A, B, dict_1, dict_2, words)
    else:
        index = B[:,dict_1[words[i]]].reshape((B.shape[0], ))
        trans = A.transpose().dot(alpha[i-1])
        alpha_part = np.multiply(index, trans)
        alpha[i] = alpha_part
        create_alpha(i+1, len(words), alpha, pi, A, B, dict_1, dict_2, words)
        
        

def create_beta(i, end, beta, alpha, pi, A, B, dict_1, dict_2, words):
    if i == end:
        return beta
    if i == len(words):
        beta[i-1] = 1
        create_beta(i-1, 0, beta, alpha, pi, A, B, dict_1, dict_2, words)
    else: 
        index = B[:,dict_1[words[i]]]
        multi = np.multiply(beta[i], index)
        beta_part = A.dot(multi)
        beta[i-1] = beta_part
        create_beta(i-1, 0, beta, alpha, pi, A, B, dict_1, dict_2, words)
             
#Probability

def probability(words, alpha,beta,dict_2):
    likelihood = np.zeros((len(words),len(dict_2)))
    for i in range(len(words)):
        denom = np.sum(alpha[i])  
        for j in range(len(dict_2)):
            prob = (alpha[i][j]*beta[i][j])/denom
            likelihood[i][j] = prob
    return likelihood
        

def classification(likelihood,keys):
    pred_labels = []
    for i in range(len(likelihood)):
        num = np.argmax(likelihood[i])
        letter = keys[num]
        pred_labels.append(letter)
    return pred_labels
        

def predict(words, pred_labels):
    for i in range(len(words)):
        words[i] = words[i] + "_" + pred_labels[i]
    return words

def accuracy(prediction_labels_acc, validation):
    
    correct = 0
    total = 0
    
    for i in range(len(prediction_labels_acc)):
        for j in range(len(prediction_labels_acc[i])):
            if prediction_labels_acc[i][j] == validation[i][j]:
                correct = correct + 1
                total = total + 1
            else:
                total = total + 1
    
    accuracy = correct/total
    
    return accuracy

    
def listToString(s):  
    
    # initialize an empty string 
    str1 = ""  
    
    for i in range(len(s)):
        if i == (len(s)-1):
            str1 += s[i]
        else:
            str1 += s[i] + " "
    
    # return string   
    return str1  

    
def main():  

    import numpy as np
    import csv

    #File
    #predicted_import = open("full_predicted.txt", "r").read().splitlines()
    B = np.array(list(csv.reader(open(emit, "r"), delimiter = " "))).astype(float)
    pi = np.array(list(csv.reader(open(pri, "r"), delimiter = " "))).astype(float)
    A = np.array(list(csv.reader(open(trans, "r"), delimiter = " "))).astype(float)
    
    #train_import = open("full_train.txt", "r").read().splitlines()
    #index_to_word = np.array(open("full_index_to_word.txt").read().splitlines())
    #index_to_tag = np.array(open("full_index_to_tag.txt").read().splitlines())
    #validation_import = open("full_validation.txt", "r").read().splitlines()
    
    index_to_word = np.array(open(word).read().splitlines())
    index_to_tag = np.array(open(tag).read().splitlines())
    validation_import = open(vali, "r").read().splitlines()
    
    validation = []
    
    for row in validation_import:
        line = row.split(" ")
        validation.append(line)
        
    #predicted = []
    
    #for row in predicted_import:
    #    line = row.split(" ")
    #    predicted.append(line)


    dict_1, dict_2, tag_length, word_length = create_dictionary(index_to_tag, index_to_word)
    keys = getList(dict_2)
    
    prediction_labels = []
    log_likelihood = []
    
    
    for i in range(len(validation)):
        true_labels, words = valid(validation[i])
        alpha = np.zeros((len(words),tag_length))
        create_alpha(0, len(words), alpha, pi, A, B, dict_1, dict_2, words)
        beta = np.zeros((len(words),tag_length))
        create_beta(len(words), 0, beta, alpha, pi, A, B, dict_1, dict_2, words)   
        likelihood = probability(words, alpha,beta,dict_2)
        pred_labels = classification(likelihood,keys)
        pred_words = predict(words, pred_labels)
        prediction_labels.append(pred_words)
        
        
        prob = np.log(alpha[-1,:].sum())
        log_likelihood.append(prob)
    
    #fix prediction labels into strings
    
    prediction_labels_acc = prediction_labels
    correct = accuracy(prediction_labels_acc, validation)
    
    for i in range(len(prediction_labels)):
        prediction_labels[i] = listToString(prediction_labels[i])
    
    prediction_labels = np.array(prediction_labels)
    log_likelihood = np.array(log_likelihood)
    
    avg = np.mean(log_likelihood)
    
    #prediction labels        
    with open(pred,"w+") as f:
        for row in prediction_labels:
            f.write("%s\n"%(row))
    
    #metrics output
    
    with open(met,"w") as g:
        g.write("Average Log-Likelihood: %f\n" % (avg))
        g.write("Accuracy: %f\n" % (correct))

        
    
    
 
if __name__ == '__main__':
    
    ##import appropiate packages
    import sys
    import csv
    import numpy as np
    
    
    
    vali = sys.argv[1]
    word = sys.argv[2]
    tag = sys.argv[3]
    pri = sys.argv[4]
    emit = sys.argv[5]
    trans = sys.argv[6]
    pred = sys.argv[7]
    met = sys.argv[8]
    
    main()
       
        




