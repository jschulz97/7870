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

h1_bias     =   np.array([.2, .1, .9])

h2_weights  =   np.array([[ 0,  0,  0],
                          [.1, .1, .1],
                          [.1, .1, .1],
                          [.2, .2, .2]
                          ])

h2_bias     =   np.array([0, .2, 0, -.1])

out_weights =   np.array([[1.5, 1.2,  1, 0],
                          [  0,  .8, .1, 0]
                          ])

out_bias    =   np.array([-.2, -.1])

do_plot     = True
eta         = .01    # learning rate
max_epoch   = 1     # how many epochs? (each epoch will run through all 4 data points)
err         = np.zeros((max_epoch,1))   # lets record error to plot (get a convergence plot)
v1          = np.ones((3,1))
v2          = np.ones((4,1))
o           = np.ones((2,1))

###############################################
# Epochs
###############################################
for k in range(max_epoch): 
    
    # init error
    err[k] = 0    

    for i,x in enumerate(data):
        ## forward pass
        for j in range(3):
            # layer 1 -- v1 is (3,1), x is (1,3), h1_w is (3,3) 
            v1[j] = np.dot(x, np.transpose(h1_weights[j])) + h1_bias[j] # h1
            v1[j] = tanh(v1[j])

        for j in range(4):
            # layer 2 -- v2 is (4,1), v1 is (3,1), h2_w is (4,3) 
            v2[j] = np.dot(np.transpose(v1), np.transpose(h2_weights[j])) + h2_bias[j] # h2
            v2[j] = tanh(v2[j])

        for j in range(2):
            # output layer -- o is (2,1), v2 is (4,1), out_w is (2,4)
            o[j] = np.dot(np.transpose(v2), np.transpose(out_weights[j])) + out_bias[j] # out
            o[j] = o[j]
        
        # error
        # y is (8,2), o is (2,1)
        err[k] += np.sum(((1.0/2.0) * np.power((y[i] - o.T), 2.0)))


        ## backprop
        # output layer -- (8,2)
        #delta_ow = (-1.0) * (y - o) * tanh(o,derive=True)
        delta_ow     = (-1.0) * (y - o)

        # Layer 2
        #(8,4)             (8,2)      (2,4)           (8,4)
        delta_h2     = np.dot(delta_ow , out_weights) * tanh(v2,derive=True)
        
        # Layer 1
        #(8,3)             (8,4)      (4,3)           (8,3)
        delta_h1     = np.dot(delta_h2 , h2_weights) * tanh(v1,derive=True)

        # update rule
        h1_weights  = h1_weights  - np.transpose( eta * np.dot(np.transpose(X) , delta_h1) ) #hidden layer 1
        h1_bias     = h1_bias - np.transpose( eta * np.dot(np.ones((1,8)) , delta_h1) )
        h2_weights  = h2_weights  - np.transpose( eta * np.dot(np.transpose(v1) , delta_h2) ) #hidden layer 2
        h2_bias     = h2_bias - np.transpose( eta * np.dot(np.ones((1,8)) , delta_h2) )
        out_weights = out_weights - np.transpose( eta * v2.T.dot( delta_ow ) ) #output layer
        out_bias    = out_bias - np.transpose( eta * np.dot(np.ones((1,8)), delta_ow))
    

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