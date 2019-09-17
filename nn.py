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


# print('hidden layer 1, neuron 1 weights')
# print(n1_w)
# print('hidden layer 1, neuron 2 weights')
# print(n2_w)
# print('hidden layer 2, neuron 1 weights')
# print(n3_w)

###############################################
# Epochs
###############################################
eta = 5.0 # learning rate
err_break = 0.001 # stop when below this error
max_epoch = 10 # how many epochs? (each epoch will run through all 4 data points)
err = np.zeros((max_epoch,1)) # lets record error to plot (get a convergence plot)

for k in range(max_epoch): 
    
    # init error
    err[k] = 0    
    
    # random shuffle of data each epoch?
    #inds = np.random.permutation(inds)
            
    # forward pass
    # layer 1
    v1 = np.dot(X, np.transpose(h1_weights)) + h1_bias # h1
    v1 = sigmoid(v1)
    # v1 is (8,3)  

    # layer 2
    v2 = np.dot(v1, np.transpose(h2_weights)) + h2_bias # h2
    v2 = sigmoid(v2)
    # v2 is (8,4) 

    # output layer
    oo = np.dot(v2, np.transpose(out_weights)) + out_bias # out
    o  = sigmoid(oo) # hey, result of our net!!!
    # o is (8,2)
    
    # error
    err[k] = np.sum(((1.0/2.0) * np.power((y - o), 2.0)))
            
    # backprop time folks!!!
    
    # output layer, our delta is (delta_1 * delta_2)
    delta_1 = (-1.0) * (y - o) #(8,2)
    delta_2 = sigmoid(o,derive=True) # note how I called it, I passed o=sigmoid(oo) # (8,2)
    
    # now, lets prop it back to the weights
    #delta_ow = np.ones((8, 3, 1))
    
    # format is
    #  delta_index  =         input to final neuron    . (Err derivative * Sigmoid derivative)
    #(4,2)          =         (8,4)T                   . (8,2)      
    delta_ow        = np.dot( np.transpose(v2)         , (delta_1        * delta_2)                   )

    # Layer 2
    delta_4 = sigmoid(v2,derive=True) #(8,4)
    #error_l =         input            this Sig der        error from output         weight to output neuron
    #(4,3)   =        [(8,4)          * (8,4)]T     .    [ (8,2)          .           (4,2)T           ]
    #delta_h2 = np.dot(np.transpose(v1 * delta_4) , np.dot(delta_1*delta_2 , np.transpose(out_weights)) )
    delta_h2 = np.dot(np.transpose(delta_4) , np.dot(delta_1*delta_2 , np.transpose(out_weights)) )
    
    #(4,3)      (8,4)                     (8,2)     (4,2)T
    delta_h2 = sigmoid(v2,derive=True) * delta_1*delta_2 . np.transpose(out_weights)
    
    # Layer 1
    delta_3 = sigmoid(v1,derive=True) #(8,3)
    #error_l =         input           this Sig der       error from output          weight to output neuron
    #(3,3)   =        [(8,3)         * (8,3)]T     .    [ (8,4)          .           (3,4)T           ]
    #delta_h1 = np.dot(np.transpose(X * delta_3) , np.dot(delta_4         , np.transpose(h2_weights)) )
    delta_h1 = np.dot(np.transpose(delta_3) , np.dot(delta_4         , np.transpose(h2_weights)) )

    # update rule
    h1_weights  = h1_weights  - eta * delta_h1 #hidden layer 1
    h2_weights  = h2_weights  - eta * delta_h2 #hidden layer 2
    out_weights = out_weights - eta * delta_ow #output layer
    

# plot it        
plt.plot(err)
plt.ylabel('error')
plt.xlabel('epochs')
plt.show()