# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 19:16:29 2021

@author: User
"""
import numpy as np
import pandas as pd 
from  NeuronNetwork import relu,softmax,tanh,sigmoid,LeakyRelu,bin_entropy_loss
from  NeuronNetwork import NeuronNetwork




# load the data
train_data = pd.read_csv('DATA_TRAIN.CSV',header=None)
valid_data = pd.read_csv('DATA_valid.CSV',header=None)
# name labels
train_data.columns = ['x','y','label']
valid_data.columns = ['x','y','label']
#shuffle data
train_data = train_data.sample(frac=1)
valid_data = valid_data.sample(frac=1)
# divide to data and labels
X = train_data.loc[:, train_data.columns != 'label'].to_numpy()
y = train_data.loc[:, train_data.columns == 'label'].to_numpy()
X_valid = valid_data.loc[:, valid_data.columns != 'label'].to_numpy()
y_valid = valid_data.loc[:, valid_data.columns == 'label'].to_numpy()


nn1 = NeuronNetwork(input_size = 2,output_size = [1], hidden_layers= [8,4], 
                 xavier = False,cost_function = bin_entropy_loss,activations=[relu,tanh],
                 activation_func = tanh ,output_activation = sigmoid,optimizer='SGD') # create the NN model with default features


nn2 = NeuronNetwork(hidden_layers= [48,24,12,6], 
                 xavier = True,cost_function = bin_entropy_loss,
                 activation_func = relu ,output_activation = sigmoid,optimizer='SGD',regularization='l2',lamda=0.04,moment=0.2,decay=0.04) # create the NN model with default features

nn1.train(X,y,epochs=1,lr=0.05,batch_size=0)
nn1.plot_loss()



train_pred = nn1.predict(X)
test_pred = nn1.predict(X_valid)

print("Train accuracy is {}".format(nn1.accuracy(y, train_pred)))
print("Test accuracy is {}".format(nn1.accuracy(y_valid, test_pred)))

# nn2 trains with mini-batch learning over 1000 epochs
nn2.train(X,y,epochs=1,lr=0.01,batch_size=20)
nn2.plot_loss()



train_pred = nn2.predict(X)
test_pred = nn2.predict(X_valid)

print("Train accuracy is {}".format(nn2.accuracy(y, train_pred)))
print("Test accuracy is {}".format(nn2.accuracy(y_valid, test_pred)))