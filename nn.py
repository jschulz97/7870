# libs
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from IPython.display import clear_output
import sys
import random

# our nonlinear Sigmoid function (including its derivative)
#  note: I let a=1 in the Sigmoid
def sigmoid(x, derive=False): # x is the input, derive is do derivative or not
    if derive: # ok, says calc the deriv?
        return x * (1.0 - x) # note, you might be thinking ( sigmoid(x) * (1 - sigmoid(x)) )
                           # depends on how you call the function
    return ( 1.0 / (1.0 + np.exp(-x)) )


##
# tanh
def tanh(x, derive=False):
    if derive:
        return np.power(1/np.cosh(x),2)
    return np.tanh(x)


# data set
X = np.array([
    [0,0,0],
    [0,0,1],
    [0,1,0],
    [0,1,1],
    [1,0,0],
    [1,0,1],
    [1,1,0],
    [1,1,1]
]) 

# labels
y = np.array([[1,0], 
              [0,1], 
              [0,1],
              [1,0],
              [0,1],
              [1,0],
              [1,0],
              [1,0]
              ])

# weights with random numbers
h1_weights  =   np.array([[.1, .2, .3],
                          [.1, .1, .1],
                          [.3, .3, .3]
                          ])

h1_bias     =   np.array([[.2], [.1], [.9]])

h2_weights  =   np.array([[ 0,  0,  0],
                          [.1, .1, .1],
                          [.1, .1, .1],
                          [.2, .2, .2]
                          ])

h2_bias     =   np.array([[0], [.2], [0], [-.1]])

out_weights =   np.array([[1.5, 1.2,  1, 0],
                          [  0,  .8, .1, 0]
                          ])

out_bias    =   np.array([-.2, -.1])


do_plot     = False
eta         = .01   # learning rate
max_epoch   = 1     # how many epochs? (each epoch will run through all 4 data points)
err         = np.zeros((max_epoch,1))   # lets record error to plot (get a convergence plot)


###############################################
# Epochs
###############################################

for k in range(max_epoch): 
    
    # init error
    err[k] = 0    


    ## forward pass
    # layer 1 -- v1 is (8,3) 
    v1 = np.dot(X, np.transpose(h1_weights)) + h1_bias.T # h1
    v1 = tanh(v1)

    # layer 2 -- v2 is (8,4) 
    v2 = np.dot(v1, np.transpose(h2_weights)) + h2_bias.T # h2
    v2 = tanh(v2)

    # output layer -- o is (8,2)
    oo = np.dot(v2, np.transpose(out_weights)) + out_bias.T # out
    o = oo
    
    # error
    err[k] = np.sum(((1.0/2.0) * np.power((y - o), 2.0))) / 8


    ## backprop
    # output layer -- (8,2)
    delta_ow = (-1.0) * (y - o) * tanh(o,derive=True)

    # Layer 2
    #(8,4)             (8,2)      (2,4)           (8,4)
    delta_h2 = np.dot(delta_ow , out_weights) * tanh(v2)
    delta_h2bias = np.dot(delta_ow , out_weights) * tanh(v2)
    
    # Layer 1
    #(8,3)             (8,4)      (4,3)           (8,3)
    delta_h1 = np.dot(delta_h2 , h2_weights) * tanh(v1)
    delta_h1bias = np.dot(delta_h2 , h2_weights) * tanh(v1)

    # update rule
    h1_weights  = h1_weights  - np.transpose( eta * np.dot(np.transpose(X) , delta_h1) ) #hidden layer 1
    h1_bias     = h1_bias - np.transpose( eta * np.dot(np.ones((1,8)) , delta_h1bias) )
    h2_weights  = h2_weights  - np.transpose( eta * np.dot(np.transpose(v1) , delta_h2) ) #hidden layer 2
    h2_bias     = h2_bias - np.transpose( eta * np.dot(np.ones((1,8)) , delta_h2bias) )
    out_weights = out_weights - np.transpose( eta * v2.T.dot( delta_ow ) ) #output layer
    

print("\n\nHidden Layer 1 Weights:\n\n",h1_weights)
print("\n\nHidden Layer 1 Biases:\n\n",h1_bias)

print("\n\nHidden Layer 2 Weights:\n\n",h2_weights)
print("\n\nHidden Layer 2 Biases:\n\n",h2_bias)

print("\n\nOutput Weights:\n\n",out_weights)
print("\n\nOutput Biases:\n\n",out_bias)

# plot it
if(do_plot):   
    plt.plot(err)
    plt.ylabel('error')
    plt.xlabel('epochs')
    plt.show()