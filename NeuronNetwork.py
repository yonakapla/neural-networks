# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 16:37:18 2021

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

def relu(X):
   return np.maximum(0,X)

def LeakyRelu(X):
  alpha = 0.01
  mask = (X>0)*X
  mask1 =(X<=0)*alpha*X
  return mask+mask1

def sigmoid(X):
   return 1/(1+np.exp(-X))

def softmax(X):
    expo = np.exp(X)
    expo_sum = np.sum(np.exp(X))
    return expo/expo_sum   

def tanh(X):
    return (np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))

def dsigmoid(X):
    return X*(1-X)

def dtanh(X):
    return 1- np.square((np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X)))
    
def xavier (inp,out):
    return np.random.normal(loc=0,scale=(1/inp),size=(inp,out))


def drelu(x):
    mask = (x>0)*1.0
    mask2 = (x<=0)*0.0
    return mask+mask2

def dLRelu(x):
  alpha = 0.01
  mask = (x>0)*1.0
  mask2 = (x<=0)*alpha
  return mask+mask2

def eta(x):
      ETA = 0.0000000001
      return np.maximum(x, ETA)
  
def bin_entropy_loss(y, yhat):
        '''
        calculate binary cross entropy
        '''
        nsample = len(y)
        yhat_inv = 1.0 - yhat
        y_inv = 1.0 - y
        yhat = eta(yhat) ## clips value to avoid NaNs in log
        yhat_inv = eta(yhat_inv)
        
        loss = (1/nsample)*np.sum(-y*np.log(yhat)-(y_inv)*np.log(yhat_inv))
        
        return loss

def MSE (y,yhat):
    nsample = len(y)
    
    loss = (1/nsample*2)*np.square(y-yhat)
    return loss

class NeuronNetwork():
    '''
    A multi layer neural network (with binary output- in order to have more classes define different loss function)
    input_size determines the input layer dimension. output_size determines the output layer dimension. 
    hidden_layers determines the number of layers and neurons in each layer(e.g [8,8,8] will be a network with three hidden layers of 8 neurons each)
    learning rate determines the learning rate of the train function -- moved to train function
    iteration determines how many times to perform the training (epochs) -- moved to train function (new name--> epochs)
    xavier determines whether to implement xavier's initialization of the weights, set to False
    activations defines the activation function between each layer, must be equal to the amount of hidden layers, if not specified will use the activation func across all layers
    moment determines the value of the moment element
    regularization determines whether to implement l1,l2 regularization choose from ['l1','l2','']
    lamda detemines lambda value when using regularization, when no reg is chosen, must be set to 0.0
    optimizer determines the optimizer method during backprop - avail: ['SGD','adagrad','RMSprop']
    activation_func determines which activation function to perform accross ALL layers, it uses the associated derivative during backprop
    loss_function defines the cost function to use when calulating and backproping the loss, avail ['MSE',binary cross entropy], default 'bin_entropy_loss'
    '''
        
    def __init__(self,input_size = 2,output_size = [1], hidden_layers= [8], 
                 xavier = False,cost_function = bin_entropy_loss,activations = [tanh],
                 activation_func = tanh ,output_activation = sigmoid ,moment = 0.0,regularization = '',lamda = 0.0,optimizer='SGD',decay = 0.02):
                 
        self.weights = {}
        self.forward_weights = {}
        self.bias = {}
        self.learning_rate = 0
        self.decay = decay
        self.loss = []
        self.sample_size = None
        self.layers = hidden_layers
        self.layers+= output_size
        self.activation_func = activation_func
        self.output_activation = output_activation
        self.input_size = input_size
        self.output_size = output_size
        self.net_depth = len(self.layers)
        self.xavier = xavier
        self.moment = moment
        self.regularization = regularization
        self.lamda = lamda
        self.activation = []
        self.vel = {}
        self.cache={}
        self.opt = optimizer
        self.loss_function = cost_function
        self.X = None
        self.y = None
        if len(activations) ==1 and self.net_depth>2:
            self. activations = [activation_func]*(self.net_depth-1) + [output_activation]
        else:
            self. activations = activations + [output_activation]
        
                
    def init_weights(self):
        '''
        Initialize the weights from a random normal distribution or perform xavier init when normalization = true
        '''
        
        net_size = len(self.layers)
        prev_layer = self.input_size
        if  not xavier:
            for layer,layer_num in zip(self.layers,range(len(self.layers))):
    
              self.weights[f'{layer_num}W'] = np.random.randint(prev_layer,layer)
              self.bias[f'{layer_num}b'] = np.random.randint(layer,)
              prev_layer = layer
              
        else: # if initialized then xavier
            for layer,layer_num in zip(self.layers,range(len(self.layers))):
    
              self.weights[f'{layer_num}W'] = xavier(prev_layer,layer)
              self.bias[f'{layer_num}b'] = np.zeros(layer,)
              prev_layer = layer
              
        if self.moment>0: # if we have moment than calc
            for layer,layer_num in zip(self.layers,range(len(self.layers))):
              self.vel[f'{layer_num}W'] = np.zeros((prev_layer,layer))
              prev_layer = layer
              
        if self.opt!='SGD': # if we are not using 'SGD'
            for layer,layer_num in zip(self.layers,range(len(self.layers))):
              self.cache[f'{layer_num}W'] = np.zeros((prev_layer,layer))
              prev_layer = layer
              

    def dabs(self,x):# calculate derivative of abs func
        mask = (x>0)*1.0
        mask1 = (x<=0)*-1.0
        return mask1+mask


    def eta(self, x):
      ETA = 0.0000000001
      return np.maximum(x, ETA)


    def sigmoid(self,Z):
        '''
        The sigmoid function takes in real numbers in any range and 
        squashes it to a real-valued output between 0 and 1.
        '''
        return 1/(1+np.exp(-Z))

    def bin_entropy_loss(self,y, yhat):
        '''
        calculate binary cross entropy
        '''
        nsample = len(y)
        yhat_inv = 1.0 - yhat
        y_inv = 1.0 - y
        yhat = self.eta(yhat) ## clips value to avoid NaNs in log
        yhat_inv = self.eta(yhat_inv)
        
        loss = (1/nsample)*np.sum(-y*np.log(yhat)-(y_inv)*np.log(yhat_inv))
        
        return loss

    def l2_reg(self,loss):
      # adds a the lambda elemnt to the loss function as in error = error + lamda/(2+N)*(sum(W^2)) 
        reg_cost = 0.0
        nsample = len(self.y)
        
        for (name_W,W) in (self.weights.items()):
            reg_cost += np.sum(np.square(W))
        
        reg_cost = loss + self.lamda/(2+nsample)*reg_cost
            
        return reg_cost

    def l1_reg(self,loss):
      # adds a the lambda elemnt to the loss function as in error = error + lamda/(N)*(sum(|W|)) 
        reg_cost = 0.0
        nsample = len(self.y)
        
        for (name_W,W) in (self.weights.items()):
            reg_cost += np.sum(np.abs(W))
        
        reg_cost = loss + (self.lamda*reg_cost)/nsample
            
        return reg_cost

    def forward_propagation(self):
        '''
        Performs the forward propagation
        '''
        inp = self.X
        for (name_W,W),(b_name,b),activation_f in zip(self.weights.items(),self.bias.items(),self.activations):
          self.forward_weights[f'{name_W}'] = inp.dot(W)+ b
          output = self.forward_weights[f'{name_W}']
          #self.activation.append(self.activation_func(output))
          self.activation.append(activation_f(output))
          inp = self.activation[-1]
          
        

        # sigmoid the output
        y_hat = inp #self.output_activation(output)
        # discard last activation 
        self.activation.pop()
        #calc loss
        loss = self.loss_function(self.y,y_hat)
        # 
        if self.regularization == 'l2':
            loss = self.l2_reg(loss)
        elif self.regularization == 'l1':
            loss = self.l1_reg(loss)


        return y_hat,loss

    def back_propagation(self,yhat):
        '''
        Computes the derivatives and update weights and bias according to the specified optimizer, regularization and activations.
        '''
        # verifies that if no regularization then the lambda element will zero (no effect on backprop)
        l1=0.0
        l2=0.0
        sums = 0.0
        if self.regularization == 'l2':
          for w in self.weights.values():
            sums+=np.sum(w)
          l2=1.0
        elif self.regularization == 'l1': 
          for w in self.weights.values():
            sums+=np.sum(self.dabs(w))
          l1= 1.0
        
        # calc y_hat and yhat inv    
        y_inv = 1 - self.y
        yhat_inv = 1 - yhat
        

        # derivative of loss function
        if self.loss_function == bin_entropy_loss:
            dl_wrt_yhat = np.divide(y_inv, self.eta(yhat_inv)) - np.divide(self.y, self.eta(yhat)) #+ sums/len(yhat)*self.lamda*l2 + sums*self.lamda*l1
        elif self.loss_function == MSE:
            dl_wrt_yhat = (1/len(yhat))*(self.y-yhat)
            
        
        # derivative of sigmoid 
        if self.output_activation == sigmoid:
          # y_hat is already sigmoized
          dl_wrt_out = yhat * (yhat_inv)

        elif self.output_activation == softmax:
          # yhat is already softmaxed
          dl_wrt_out = yhat * (yhat_inv)
        
        elif self.output_activation == tanh:
          dl_wrt_out = 1-yhat**2
        
        # first backprop element 
        dl_wrt_z2 = 1/len(yhat)*dl_wrt_yhat * dl_wrt_out
        
        backproped = dl_wrt_z2

        #backprop over all layers
        for (name_W,W),(name_b,b),cntr in sorted(zip(self.weights.items(),self.bias.items(),range(self.net_depth)),reverse= True):
          
          dactivation = backproped.dot(W.T)       
         
          if self.activation != list(): # pops the results of the activation function
            activ_res = self.activation.pop().T
            
          else:# if in first layer use the input
            activ_res = self.X.T
            
        # calc dw/dx with regularizations and for bias
        
          dl_wrt_w2 = activ_res.dot(backproped) + (l2*(1/len(yhat))*self.lamda*W) + (l1*self.lamda*self.dabs(W)) # add regularization if lamda !=0
          dl_wrt_b2 = np.sum(backproped, axis=0, keepdims=True)
          
          if self.opt == 'adagrad':
              self.cache[f'{cntr}W'] = self.cache[f'{cntr}W'] + dl_wrt_w2**2
              dl_wrt_w2 = self.learning_rate*dl_wrt_w2/(np.sqrt(self.cache[f'{cntr}W'])+1e-8)
          
          elif self.opt == 'RMSprop': # adds the adaptive gradient element 
              self.cache[f'{cntr}W'] = self.decay*self.cache[f'{cntr}W'] +(1-self.decay)*dl_wrt_w2**2
              dl_wrt_w2 = self.learning_rate*dl_wrt_w2/(np.sqrt((self.cache[f'{cntr}W'])+1e-8))

          
          if self.moment>0:# adds moment element velocity = moment*velocity - eta*dW, update rule = W = W + velocity
              self.vel[f'{cntr}W'] = self.moment*self.vel[f'{cntr}W']-self.learning_rate*dl_wrt_w2
              self.weights[name_W] = self.weights[name_W] + self.vel[f'{cntr}W']
          else:
              self.weights[name_W] = self.weights[name_W] - self.learning_rate*dl_wrt_w2
                  
          # update bias, the update is always by the SGD rule as it can be easily compencated by the update in weights.
          self.bias[name_b] = self.bias[name_b] - self.learning_rate*dl_wrt_b2

          # calc next backproped
          if cntr !=0:
            if self.activations[cntr-1] == tanh:#self.activation_func==tanh:
              dactiv = dtanh
            elif self.activations[cntr-1] == relu:#self.activation_func == relu:
              dactiv = drelu
            elif self.activations[cntr-1] == sigmoid:#self.activation_func == sigmoid:
              dactiv = dsigmoid
            elif self.activations[cntr-1] == LeakyRelu:#self.activation_func == LeakyRelu:
              dactiv = dLRelu 
            else:
              print('activation function not supported')

              
            # calc next bacproped item
            backproped = dactivation  * dactiv(self.forward_weights[f'{cntr-1}W'])


    def train(self, X, y,batch_size = 0,lr=1e-4,epochs = 1,learning_rate_decay = 1.0):
        '''
        Train the network on the given data, supports batch learning
        
        '''
        self.learning_rate = lr
          
        self.init_weights() #initialize weights and bias
        
        if batch_size > 0:
          for i in range(epochs):
            batch_loss= []
            pos = 0
            self.learning_rate = self.learning_rate*learning_rate_decay
            for batch in range(batch_size,len(X),batch_size):
              
              self.X = X[pos:batch]
              self.y = y[pos:batch]
              yhat, loss = self.forward_propagation()
              self.back_propagation(yhat)
              batch_loss.append(loss)
              pos = batch
            self.loss.append(np.mean(batch_loss))
        else:
          self.X = X
          self.y = y
          for i in range(epochs):
            yhat, loss = self.forward_propagation()
            self.back_propagation(yhat)
            self.loss.append(loss)



       
    def predict(self, X):
        '''
        Predicts the output of given test data
        '''
        inp = X
        # propogate forward
        for (name_W,W),(b_name,b) in zip(self.weights.items(),self.bias.items()):
          output = inp.dot(W)+ b
          inp = self.activation_func(output)
          
        pred = self.output_activation(output)
        return np.round(pred)

    def accuracy(self, y, yhat):
        '''
        Calculates the accuracy of the network's predictions
        '''
        acc = int(np.sum(y == yhat) / len(y) * 100)
        return acc


    def plot_loss(self):
        '''
        Plots the loss curve
        '''
        plt.plot(self.loss)
        plt.xlabel("Epochs")
        plt.ylabel("logloss")
        plt.title("Loss curve for training")
        plt.show()

