#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 19:15:36 2020

@author: ssriskanda
"""

#function to open training, test, and validation file        
def openfile(train, test, valid):
    #train: training data
    #test: testing data
    #valid: validation data
    
    with open(train) as file:
        train = np.array(list(csv.reader(file, delimiter = '\t')))

    with open(test) as file:
        test = np.array(list(csv.reader(file, delimiter = '\t')))

    with open(valid) as file:
        valid = np.array(list(csv.reader(file, delimiter = '\t')))
    
    return train, test, valid

#functinos used to create dictionary
def open_dict(dictionary):
    with open(dictionary, "r") as file: #used to open the dictionary file
        temp = list(csv.reader(file, delimiter = '\n'))
        
    return temp #returns temporary format of lists of lists that will be used for next function


def create_dict(temp):
    #temp: temporary list of lists created by the open_dict 
    
    dic = [] #temporary place to hold information 
    dictionary = {} #final outcome

    #for each item, split the list and append it to a temporary dict list
    for i in range(len(temp)):
        for item in temp[i]:
            item = item.split(" ")
            dic.append(item)
    
    #for each item in dic, set the index as the key and the value as 1/0
    for i in range(len(dic)):
            dictionary[dic[i][0]] = dic[i][1]
    
    #keep number of keys in mind to be used to calculate the parameters vector
    keys = dictionary.keys()
    params = [0] * (len(keys) + 1) #length of possible number of keys and 1 for bias
    
    return dictionary,keys, params



def parse(data):
    inputs = [] #feature vectors
    outputs = [] #outcomes
    
    #for each observation in data, take the outcome for that observation and add it to the list
    for i in data:
        outputs.append(int(i[0]))
        
        #temporary row to make list of lists for inputs
        temp_row = []
        
        for j in range(1,len(i)):
            line = i[j].split(":")
            temp_row.append(int(line[0]))
        temp_row.append(0) #add bias term to the vector
        inputs.append(temp_row) #add it to the final features vector
        
    return inputs, outputs 
    
        
        
#function for SGD
def sgd(alpha, params, inputs, outputs, epoch):
    #alpha: alpha provided in main() function
    #params: theta vector that would be changed
    #inputs: feature vector
    #outputs: potential outcome for each observation
    #epoch: maximum number of epochs needed 
    
    count = 0 #count number of epochs completed at this point
    length = len(inputs) #total number of observations 
    
    while count < epoch:
        for i in range(len(inputs)):
            dot = 0 #intializing the dot sum to 0 
            
            for j in range(len(inputs[i])):
                dot += params[inputs[i][j]] #calculate the dotsum of theta and x 
                
            for j in range(len(inputs[i])):
                #calculate for logistic regression to change the theta
                params[inputs[i][j]] += (alpha/length)*(outputs[i] - (math.exp(dot)/(1+math.exp(dot))))
                
        count += 1 #one round complete      
    return params


#prediction function
def predict(data, params):
    labels = []
    features, true_ouputs  = parse(data) #obtain feature importation and the true outputs for calculating the error
    
    for i in range(len(features)):
        dot = 0
        for k in range(len(features[i])):
            dot += params[features[i][k]]
            
        predic = math.exp(dot)/(1+math.exp(dot)) #calculate probability
        
        if predic >= 0.5: #given in the hw for classification rule
            labels.append(1)
        else:
            labels.append(0)
    
    return labels,true_ouputs


#calculate the error rate based on the identification and the true output
def errorcal(true_ouputs, labels):
    error = 0
    
    for i in range(len(labels)):
        if true_ouputs[i] == labels[i]:
            error = error
        else:
            error += 1
    
    #calculations
    error = (error)/len(labels)        
    
    return error   




def likelihood(inputs, outputs, alpha, epoch, keys, params):
    #input: data feature
    #output: data outcomes
    #alpha: alpha that is provied
    #epoch: number of epochs asked for
    #key: dictionary keys to build theta
    
    count = 0 #count number of epochs
    length = len(outputs) #number of observations
    average = [] #empty list to gather all 
    theta = []
    
    while count < epoch:
        
        probability = []
        
        for i in range(len(inputs)):
            dot = 0
            
            for j in range(len(inputs[i])):
                dot += params[inputs[i][j]]
                
            for j in range(len(inputs[i])):
                params[inputs[i][j]] += (alpha/length) * (outputs[i] - (math.exp(dot)/(1+math.exp(dot))))
    
        loglike = -1*outputs[i]*dot + np.log(1+ math.exp(dot))
        probability.append(loglike)  
            
        theta.append(params)
        avg = (1/length) * sum(probability)
        average.append(avg)
        count+=1

    return average, theta

def validLikelihood(inputs, outputs, theta, alpha, epoch):
        
    count = 0 
    length = len(outputs)
    average = []
    
    while count < epoch:
        for tet in theta:
            for i in range(len(inputs)):
                dot = 0
                probability = []
            
            for j in range(len(inputs[i])):
                tet[inputs[i][j]] += (alpha/length) * (outputs[i] - (math.exp(dot)/(1+math.exp(dot))))
                
                
            loglike = -1*outputs[i]*dot + np.log(1+ math.exp(dot))
            probability.append(loglike)

            avg = 1/length * sum(probability)
            average.append(avg)
            count+=1  
        
    return average
    


        
def main(): 
        
    import csv
    import numpy as np
    import math

    test = "/Users/ssriskanda/Documents/Carnegie_Mellon/Second_Year/First_Semester/10601/hw4/handout/largeoutput/model1_formatted_test.tsv"
    train = "/Users/ssriskanda/Documents/Carnegie_Mellon/Second_Year/First_Semester/10601/hw4/handout/largeoutput/model1_formatted_train.tsv"
    valid = "/Users/ssriskanda/Documents/Carnegie_Mellon/Second_Year/First_Semester/10601/hw4/handout/largeoutput/model1_formatted_valid.tsv"


    with open("dict2.txt", "r") as file:
        temp = list(csv.reader(file, delimiter = '\n'))
     
    
    train, test, valid = openfile(train, test, valid)
    
    temp = open_dict(diction)
  
    dictionary, keys, params = create_dict(temp)
    
    inputs, outputs = parse(train)
    
    #epoch = num_epoch
    epoch = 200
    alpha = 0.1   
    new_params = sgd(alpha, params, inputs, outputs, epoch)
    #print(np.unique(new_params))
    
    train_labels,train_true_ouputs = predict(train, new_params)
    train_error = errorcal(train_labels,train_true_ouputs)
    
    
    
    test_labels,test_true_ouputs = predict(test, new_params)
    test_error = errorcal(test_labels,test_true_ouputs)
    
   
    #written part
    dictionary, keys, params = create_dict(temp) #want to clear parameters
    train_mean, train_theta = likelihood(inputs, outputs, alpha, epoch, keys, params)
    train_mean = np.array(train_mean)
    

    valid_inputs, valid_ouputs = parse(valid)
    valid_mean = validLikelihood(valid_inputs,valid_ouputs , train_theta, alpha, epoch)
    #dictionary, keys, params = create_dict(temp)

    #valid_mean = likelihood(valid_inputs, valid_ouputs, alpha, 200, keys, params) 
    valid_mean = np.array(valid_mean)
    
    valid_mean = np.transpose(valid_mean)
    train_mean = np.transpose(train_mean)
    
    data = np.column_stack((valid_mean,train_mean))
    
   import pandas as pd 
   pd.DataFrame(data).to_csv("/Users/ssriskanda/Documents/Carnegie_Mellon/Second_Year/First_Semester/10601/model1_2.csv")
    
    
    #print(train_error)
    #print(test_error)

    f = open(train_out, "w+")
    for row in train_labels:
        f.write("%d\n"%(row))
        
    g = open("test_out.labels", "w+")
    for row in test_labels:
        g.write("%d\n"%(row))      
    
    h = open(metrics_out,"w+" )
    h.write("error(train): %f\n"%(train_error))
    h.write("error(test): %f\n"%(test_error))
        
    

if __name__ == '__main__':
    
    ##import appropiate packages
    import sys
    import csv
    import numpy as np
    import math
    
    trai = sys.argv[1]
    vali = sys.argv[2]
    tes = sys.argv[3]
    diction = sys.argv[4]
    train_out = sys.argv[5]
    test_out = sys.argv[6]
    metrics_out = sys.argv[7]
    num_epoch = int(sys.argv[8])
    
    
    main()



    


