import os
import sys
import random
import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, precision_score, classification_report, plot_confusion_matrix, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC as SVM

import warnings
warnings.filterwarnings('ignore')


# initialize population
def initialize(num_agents, num_features):
    # define min and max number of features
    min_features = int(0.3 * num_features)
    max_features = int(0.6 * num_features)

    # initialize the agents with zeros
    agents = np.zeros((num_agents, num_features))

    # select random features for each agent
    for agent_no in range(num_agents):

        random.seed(time.time() + agent_no)

        num = random.randint(min_features,max_features)
        pos = random.sample(range(0,num_features - 1),num)

        for idx in pos:
            agents[agent_no][idx] = 1 

    return agents


# sort agents based on fitness values
def sort_agents(agents, obj, data):
    # sort the agents according to fitness
    train_X, val_X, train_Y, val_Y = data.train_X, data.val_X, data.train_Y, data.val_Y
    (obj_function, weight_acc) = obj

    # if there is only one agent
    if len(agents.shape) == 1:
        num_agents = 1
        fitness, acc = obj_function(agents, train_X, val_X, train_Y, val_Y, weight_acc)
        return agents, fitness, acc

    # for multiple agents
    else:
        num_agents = agents.shape[0]
        fitness = np.zeros(num_agents)
        acc = np.zeros(num_agents)
        for id, agent in enumerate(agents):
            fitness[id], acc[id] = obj_function(agent, train_X, val_X, train_Y, val_Y, weight_acc)
        idx = np.argsort(-fitness)
        sorted_agents = agents[idx].copy()
        sorted_fitness = fitness[idx].copy()
        sorted_acc = acc[idx].copy()

    return sorted_agents, sorted_fitness, sorted_acc


# display details of population
def display(agents, fitness, acc, agent_name='Agent'):
    # display the population
    print('\nNumber of agents: {}'.format(agents.shape[0]))
    print('\n------------- Best Agent ---------------')
    print('Fitness: {}'.format(fitness[0]))
    print('Accuracy: {}'.format(acc[0]))
    print('Number of Features: {}'.format(int(np.sum(agents[0]))))
    print('----------------------------------------\n')

    for id, agent in enumerate(agents):
        print('{} {} - Fitness: {},Accuracy: {}, Number of Features: {}'.format(agent_name, id+1, fitness[id], acc[id], int(np.sum(agent))))

    print('================================================================================\n')


# compute accuracy of each agent
def compute_accuracy(agent, train_X, test_X, train_Y, test_Y): 
    # compute classification accuracy of the given agents
    cols = np.flatnonzero(agent)     
    if(cols.shape[0] == 0):
        return 0    

    clf = KNN()

    train_data = train_X[:,cols]
    train_label = train_Y
    test_data = test_X[:,cols]
    test_label = test_Y

    clf.fit(train_data,train_label)
    acc = clf.score(test_data,test_label)

    return acc
        

# compute fitness value of each agent
def compute_fitness(agent, train_X, test_X, train_Y, test_Y, weight_acc=0.9,dims=None):
    # compute a basic fitness measure
    if(weight_acc == None):
        weight_acc = 0.9
    weight_feat = 1 - weight_acc
    
    if dims != None:
        num_features = dims
    else:
        num_features = agent.shape[0]
    
    acc = compute_accuracy(agent, train_X, test_X, train_Y, test_Y)
    feat = (num_features - np.sum(agent))/num_features

    fitness = weight_acc * acc + weight_feat * feat
    
    return fitness, acc


def sigmoid(val):
    if val < 0:
        return 1 - 1/(1 + np.exp(val))
    else:
        return 1/(1 + np.exp(-val))


def get_trans_function(shape='s'):
        return sigmoid


def relu(X): 
    return np.maximum(0,X)


# validate an agent in FS
def validate_FS(X,y,agent,clf='knn'):
    cols = np.flatnonzero(agent)     
    if(cols.shape[0] == 0):
        return 0    
    X1 = (X[:,cols]).copy() # getting selected features
    X_train,X_test,y_train,y_test =  train_test_split(X1,y,test_size=0.2,shuffle=False)

    model = KNN() if clf == 'knn' else SVM()

    model.fit(X_train,y_train)
    # acc = model.score(test_X,y_test)
    
    y_pred = model.predict(X_test)
    accuracy = np.sum(y_pred == y_test)/y_test.shape[0]

    print(f'Accuracy for the selection using {clf}: {accuracy:.6f}')
    print('-'*50)
    
    print(classification_report(y_test,y_pred,digits=4))
    print('-'*50)
    
    plot_confusion_matrix(model,X_test,y_test)
    plt.show()