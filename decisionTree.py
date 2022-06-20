#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 10:37:08 2020

@author: ssriskanda
"""

#### BUILD ALL NODES NECESSARY FOR DECISION TREE ###
class Node: 
    def __init__(self, data,feat,x_attrib_outcome,index, attribute,leftNode, rightNode, prior_attribute):
        self.attribute = attribute #name of attribute that was split on
        self.leftNode = leftNode #branch for right
        self.rightNode = rightNode #branch for left
        self.count = outcome_count(data, y_outcome)
        self.index = int(index)
        self.feat = feat
        self.prior_attribute = prior_attribute
        
        
         
class Leaf:
    def __init__(self,data,attribute,value): #for nodes that are the end of tree
        self.data = data    
        majority, count, feature = leaf_majority(data)
        self.majority = majority
        
        output = y_outcome(data)
        final_leaf_count = leaf_count(data,output)
        self.count = final_leaf_count
        self.attribute = attribute
        self.value = value
    
###################################################

#build function to detect majority in a leaf
def leaf_majority(data):
    feature,count = np.unique(data[:,-1], return_counts = True)

    feature =feature.tolist()
    count = count.tolist()
    
    if len(feature) == 1:
        majority = feature[0]
        count = str(count[0])
        return majority, count, feature
    else:

        if count[0] > count[1]:
            majority = feature[0]
        elif count[0] < count[1]:
            majority = feature[1]
        elif count[0] == count[1]:
            majority = feature[1]

        return majority, count, feature
    
def leaf_count(data, output):
    #data: data that will do a count
    #output: all potential outcomes 
    leaf_feature, leaves_count = np.unique(data[:,-1], return_counts= True)
    
    if len(leaf_feature) == 1:
        array = [0]
        if leaf_feature  == output[0]:
            leaves_count = np.column_stack((leaves_count,array))
            leaves_count = np.array(leaves_count).ravel()
        elif leaf_feature == output[1]:
           leaves_count =  np.column_stack((array,leaves_count))
           leaves_count = np.array(leaves_count).ravel()
        return leaves_count
    else:
        leaves_count = np.array(leaves_count).ravel()
        return leaves_count
    
#function to count how many of each outcome appear
def outcome_count(data, output):
    #data: data that will do a count
    #output: all potential outcomes 
    feature, y_count = np.unique(data[:,-1], return_counts= True)
    
    if len(feature) == 1:
        array = [0]
        if feature  == output[0]:
            np.column_stack((y_count,array))
        elif feature == output[1]:
            np.column_stack((array,y_count))
        
        return y_count
    else:
        return y_count
    

#opening train and test set
def open_file(train, test):
    #train: training set of the data
    #test: testing set of the data
    with open(train) as file:
        train = list(csv.reader(file, delimiter = '\t'))        
        headers = train[0] #taking headers for naming later on
        train = np.delete(train,(0), axis = 0) #drop name of headers

    
    with open(test) as file:
        test = list(csv.reader(file, delimiter = '\t'))        
        test = np.delete(test,(0), axis = 0) #drop name of headers
    
    #return training set, testing set, and the names of the attributes
    return train, test, headers

#list of potential outcomes possible in a given dataset
def y_outcome(data):
    output = np.unique(data[:,-1]) #always be -1 since it's the last column
    return output #return y_outcome

def options(data):
    x_options = np.unique(data[:,1])
    
    return x_options

#calculate entropy H(Y)
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
        if row[-1] == feature[0]:
            count_zero += 1
        else:
            count_one += 1
    count = [count_zero, count_one]
    #entropy = -np.sum((count[i]/len(data))*np.log2(count[i]/len(data)) for i in range(len(feature)))
    entropy = -sum((count[i]/len(data))*np.log2(count[i]/len(data)) for i in range(len(feature)))

    return entropy

#list of attributes information (separating the columns for comparison) 
def attribute_list(data):
    num_col = np.shape(data)[1]
    attrib_list = np.array(np.hsplit(data,num_col))
    
    outcome = attrib_list[-1]
    attrib_list = attrib_list[0:(len(attrib_list)-1)]
    
    #return attributes and all outcomes
    return attrib_list, outcome

#calculate majority vote based on the mutual_information for each attribute and identify max entrophy
def mutual_info(data, attrib_list,outcome,output):
    #data that is in question 
    #attribute_list : attribute information in a list
    #outcome: possible choices of outcome y 
    #output: actual column vector of outcome y
    
    Mutual_info_list = list() #create list of mutual information for each attribute
    
    #iterate through each attribute
    for i in range(len(attrib_list)):
        
        #for each attribute, capture type of features, and the count for each
        attribute = attrib_list[i]
        attrib_feature, attrib_count = np.unique(attribute, return_counts = True)
        
        #combine this data with outcome column
        partial_data = np.column_stack((attribute, outcome))
        
        #separate data based on the attribute feature (zero = left node, one = right node)
        zero, one = [], []
        
        for row in partial_data:
            if row[0] == attrib_feature[0]:
                zero.append(row)
            elif row[0] == attrib_feature[1]:
                one.append(row)
        
        attrib_entropy = []
        
        if len(zero) != 0:
        #for the left data set, count how many of each y outcome
            zero_prob_zero, zero_prob_one = 0,0 
        
            for row in zero:
                if row[1] == output[0]:
                    zero_prob_zero += 1
                else:
                    zero_prob_one += 1
        
        
            zero_prob_zero = float(zero_prob_zero/len(zero))
            zero_prob_one = float(zero_prob_one/len(zero))
            
            zero_prob_zero_log = np.where(zero_prob_zero>0, np.log2(zero_prob_zero), 0)
            zero_prob_one_log = np.where(zero_prob_zero>0, np.log2(zero_prob_one), 0)
            
            zero_prob_zero_entropy = np.nan_to_num(zero_prob_zero*zero_prob_zero_log)
            zero_prob_one_entropy = np.nan_to_num(zero_prob_one*zero_prob_one_log)
            
            
            zero_entropy = -(zero_prob_zero_entropy) - (zero_prob_one_entropy)
            
            attrib_entropy.append(zero_entropy)

        if len(one) != 0:
            
            #for the right data set, count how many of each y outcome
        
            one_prob_zero, one_prob_one = 0,0 
        
            for row in one:
                if row[1] == output[0]:
                    one_prob_zero += 1
                else:
                    one_prob_one += 1    
        
            one_prob_zero = float(one_prob_zero/len(one))
            one_prob_one = float(one_prob_one/len(one)) 
            
            one_prob_zero_log = np.where(one_prob_zero>0, np.log2(one_prob_zero), 0)
            one_prob_one_log = np.where(one_prob_one>0, np.log2(one_prob_one), 0)
            
            one_prob_zero_entropy = np.nan_to_num(one_prob_zero*one_prob_zero_log)
            one_prob_one_entropy = np.nan_to_num(one_prob_one*one_prob_one_log)
        
            one_entropy = -(one_prob_zero_entropy) - (one_prob_one_entropy)
            
            attrib_entropy.append(one_entropy)

        #H_x_entropy = np.sum((attrib_count[i]/len(attribute))*attrib_entropy[i] for i in range(len(attrib_feature)))        

        entropy_y = entropy(data, -1) 
        
        H_x_entropy = sum((attrib_count[i]/len(attribute))*attrib_entropy[i] for i in range(len(attrib_feature)))  

        Mutual_info = entropy_y - H_x_entropy
        
        Mutual_info_list.append(Mutual_info)

    max_entropy = max(Mutual_info_list)

    return Mutual_info_list,max_entropy

#calculate majority vote based on the mutual_information for each attribute
def majority(data, Mutual_info_list, max_entropy, headers):

    index = np.argmax(Mutual_info_list)
    
    #attribute = headers[int(index[0])]
    attribute = headers[int(index)]
                       
    #split data based on feature in index
    
    #split_feat = np.unique(data[:,int()])
    split_feat = np.unique(data[:,int(index)])
    
    left_data, right_data = [], []
    
    for row in data: 
        if row[int(index)] == split_feat[0]:
            left_data.append(row)
        else:
            right_data.append(row)
    left_data = np.array(left_data)
    right_data = np.array(right_data)
    
    all_outcome_left = []
    
    for item in left_data:
        all_outcome_left.append(item[-1])
    
    left_val,left_count = np.unique(all_outcome_left,return_counts=True) 
    
    if len(left_val) == 1:
        left_majority = left_val[0]
    else:
        if left_count[0] > left_count[1]:
            left_majority = left_val[0]
        elif left_count[0] < left_count[1]:
            left_majority = left_val[1]
        elif left_count[0] == left_count[1]:
            left_majority = max(left_val)
            
    all_outcome_right = []
    
    for item in right_data:
        all_outcome_right.append(item[-1])
    
    right_val, right_count = np.unique(all_outcome_right, return_counts = True)
    
    if len(right_val) == 1:
        right_majority = right_val[0]
    else:
        if right_count[0] > right_count[1]:
            right_majority = right_val[0]
        elif right_count[0] <= right_count[1]:
            right_majority = right_val[1]
        elif right_count[0] == right_count[1]:
            right_majority = max(right_val)
    
    return left_majority, right_majority, np.array(left_count), right_count, np.array(attribute), left_data, np.array(right_data),index, split_feat


def pure_node(data):
    outcome, count = np.unique(data[-1], return_counts = True)
    
    return outcome,count
    
    
#function to build the nodes for the tree
def build_tree(data, index, output,headers, maxDepth, currentDepth, x_attrib_outcome, prior_attribute, feat):
    
    if maxDepth <= currentDepth:
        return Leaf(data,headers[int(index)],feat)
    elif len(pure_node(data)[1]) == 0:
        return Leaf(data, headers[int(index)],feat)
    elif currentDepth >= (len(headers) - 1):
        return Leaf(data, headers[int(index)],feat)
    else:
        attrib_list, outcome = attribute_list(data)
        
        Mutual_info_list,max_entropy = mutual_info(data, attrib_list, outcome, output) #figure out max entropy
        
        if max_entropy <= 0 or max_entropy == -0:
            return Leaf(data,headers[int(index)],feat) #do not split the data
        else:
            
            currentDepth += 1
            
            #use functions to split data
            left_majority, right_majority, left_count, right_count, attribute, left_data, right_data,index, splitting_feature= majority(data, Mutual_info_list, max_entropy, headers)
            #flattens data into 1 array
            left_data = np.stack(left_data, axis = 0)
            right_data = np.stack(right_data, axis = 0)
            
            leftNode = build_tree(left_data, index, output, headers, maxDepth,currentDepth, x_attrib_outcome, attribute, feat = splitting_feature[0])
        
            #build on the rightNode
            #if len(right_count) == 0:
            #    return None
           
            rightNode = build_tree(right_data,index, output,headers, maxDepth, currentDepth, x_attrib_outcome, attribute, feat = splitting_feature[1])
             
            #return Node(np.array_str(attribute),leftNode, rightNode)
            return Node(data,feat,x_attrib_outcome, index,np.array_str(attribute),leftNode,rightNode, prior_attribute)
        
def classification(row,node, x_attrib_outcome):
    #specific row of data
    #node = the tree created
    
    #check if the node is a leaf (reached the end)
    if isinstance(node, Leaf):
        return node.majority
    if row[node.index] == x_attrib_outcome[0]:
        return classification(row, node.leftNode, x_attrib_outcome) 
    elif row[node.index] == x_attrib_outcome[1]:
        return classification(row, node.rightNode,x_attrib_outcome)
        
def classify(data,tree, x_attrib_outcome):
        
    new_data = []
    for row in data:
        empty = np.array([""])
        row = np.hstack((row,empty))
        new_data.append(row)
        
    #flattens the data into one array
    new_data = np.stack(new_data, axis = 0)
    
    for row in new_data:
        row[-1] = classification(row,tree, x_attrib_outcome)
    
    predicted_column = new_data[:,-1]
    
    return new_data, predicted_column
            
#calculate error after classification was done above
def errorcal(new_data):
    
    #error calculation for dataset
    error = 0 
    
    #training error counting loop
    for row in new_data:
        if row[-2] == row[-1]:
            error = error + 0 
        else:
            error = error + 1
    #calculations
    error = (error)/len(new_data)        
    
    return error      

#print tree 
def print_tree(node, output,currentDepth, space = "| "):
    
    new_space = space*currentDepth   
    #print('[%d %s/%d %s]'%(node.count[0],output[0],node.count[1],output[1]))    
    if isinstance(node, Leaf):
        print(new_space + '%s = %s: [%d %s/%d %s]'%(str(node.attribute),node.value,node.count[0],output[0],node.count[1],output[1]))
        return
    elif currentDepth == 0:
         print('[%d %s/%d %s]'%(node.count[0],output[0],node.count[1],output[1]))
        # Call this function recursively on the true branch
         print_tree(node.leftNode, output,currentDepth + 1, space = "| ")
        # Call this function recursively on the false branch
         print_tree(node.rightNode, output,currentDepth + 1, space = "| ")
    
    else:
         # Print the question at this node
        print(new_space + '%s = %s:[%d %s/%d %s]'%(node.prior_attribute, node.feat, node.count[0],output[0],node.count[1],output[1]))
       
        # Call this function recursively on the true branch
        print_tree(node.leftNode, output,currentDepth + 1, space = "| ")

        # Call this function recursively on the false branch
        print_tree(node.rightNode, output,currentDepth + 1, space = "| ")
             
        
def main():   
    
    #train = "/Users/ssriskanda/Documents/Carnegie_Mellon/Second_Year/First_Semester/10601/hw2/handout/education_train.tsv"
    #test = "/Users/ssriskanda/Documents/Carnegie_Mellon/Second_Year/First_Semester/10601/hw2/handout/education_test.tsv"

    
    t_train, t_test, headers = open_file(train, test) 
    
    output = y_outcome(t_train) #options for y
    x_attrib_outcome = options(t_train)
    
    attrib_list, outcome = attribute_list(t_train)
        
    Mutual_info_list,max_entropy = mutual_info(t_train, attrib_list, outcome, output) #figure out max entropy
        
    index = majority(t_train, Mutual_info_list, max_entropy, headers)[7]

    maxDepth = maximum #shows how far it can go before it stops splitting
    currentDepth = 0  #0 at the start
    prior_attribute = None
    feat = None
    
    #build tree
    tree = build_tree(t_train, index, outcome, headers,maxDepth, currentDepth, x_attrib_outcome, prior_attribute, feat)
   
    #print tree
    print(print_tree(tree,output,currentDepth))
    
    #classification for training
    #error from training set
    train_classify, train_predict = classify(t_train, tree, x_attrib_outcome)
    train_error = errorcal(train_classify)
    
    #classification for testing
    #error from test set
    test_classify, test_predict = classify(t_test, tree, x_attrib_outcome)
    test_error = errorcal(test_classify)
    
    #output for training set
    f = open(train_out, "w+") #replace this with train out
    for row in train_predict:
        f.write("%s\n"%(row))
    f.close()       
    
    #output for testing set
    g = open(test_out, "w+") #replace this with test out
    for row in test_predict:
        g.write("%s\n"%(row))
    g.close()   
    
    #output for testing     
    h = open(metrics_out, "w+")
    h.write("error(train): %f\n"%(train_error))
    h.write("error(test): %f"%(test_error))
    h.close()   
    
    
    
if __name__ == '__main__':
    
    ##import appropiate packages
    import numpy as np
    import sys
    import csv
    
    train = sys.argv[1]
    test = sys.argv[2]
    maximum = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]
    
    main()
    

    

    
    
    
    
    
    
        
        
            
   
    
   
    
   
    
        