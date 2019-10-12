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
    [0,0,0,1],
    [0,0,1,1],
    [0,1,0,1],
    [0,1,1,1],
    [1,0,0,1],
    [1,0,1,1],
    [1,1,0,1],
    [1,1,1,1]
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
h1_weights  =   np.array([[.1, .2, .3, .2],
                          [.1, .1, .1, .1],
                          [.3, .3, .3, .9]
                          ])

#h1_bias     =   np.array([.2, .1, .9])

h2_weights  =   np.array([[ 0,  0,  0, 0],
                          [.1, .1, .1, .2],
                          [.1, .1, .1, 0],
                          [.2, .2, .2, -.1]
                          ])

#h2_bias     =   np.array([0, .2, 0, -.1])

out_weights =   np.array([[1.5, 1.2,  1, 0, -.2],
                          [  0,  .8, .1, 0, -.1]
                          ])

#out_bias    =   np.array([-.2, -.1])

do_plot     = False
eta         = .1    # learning rate
max_epoch   = 1     # how many epochs? (each epoch will run through all 4 data points)
err         = np.zeros((max_epoch,1))   # lets record error to plot (get a convergence plot)
v1          = np.ones((1,4))
v2          = np.ones((1,5))
o           = np.ones((1,2))

###############################################
# Epochs
###############################################
for k in range(max_epoch): 
    
    # init error
    err[k] = 0    

    for i,x in enumerate(X):
        ## forward pass

        #(1,3)           (1,4)      (4,3)    
        #v1[:,:3] = tanh(np.dot(x,np.transpose(h1_weights)))
        for j in range(3):
            # layer 1 -- v1 is (1,4), x is (1,4), h1_w[j] is (1,4) 
            v1[0][j] = np.dot(x, np.transpose(h1_weights[j]))
            v1[0][j] = tanh(v1[0][j])
        #v1 is (1,4)
        
        #(1,4)           (1,4)      (4,4)    
        #v2[:,:4] = tanh(np.dot(v1,np.transpose(h2_weights)))
        for j in range(4):
            # layer 2 -- v2 is (1,5), v1 is (1,4), h2_w[j] is (1,4) 
            v2[0][j] = np.dot(v1, np.transpose(h2_weights[j]))
            v2[0][j] = tanh(v2[0][j])
        #v2 is (1,5)

        #(1,2)          (1,5)      (5,2)    
        #o = tanh(np.dot(v2,np.transpose(out_weights)))
        for j in range(2):
            # output layer -- o is (1,2), v2 is (1,5), out_w[j] is (1,5)
            o[0][j] = np.dot(v2, np.transpose(out_weights[j]))
            o[0][j] = o[0][j]
        #o is (1,2)
        
        # error
        # y is (8,2), o is (2,1)
        err[k] += np.sum(((1.0/2.0) * np.power((o.T - y[i]), 2.0)))

        print("\n\nActivations: "+str(i)+"\n\n")
        print(v1)
        print(v2
        print(o)
        print("\n--------------------------\n")

        ## backprop
        
        # Output layer
        #(1,2)                      (1,2)                     (1,2) 
        delta_ow     = (-1.0) * (np.reshape(y[i],((2,1))).T - o)

        # Layer 2
        #(1,4)                (1,2)           (2,4)             (1,4)
        delta_h2     = np.dot(delta_ow, out_weights[:,:4]) * tanh(v2[:,:4],derive=True)
         
        # Layer 1
        #(1,3)           (1,4)(w/o bias update) (4,4)           (1,4)
        delta_h1     = np.dot(delta_h2[:,:3], h2_weights) * tanh(v1,derive=True)

        print("\n\noutput deltas: "+str(i)+"\n\n")
        print(delta_ow)
        print(delta_h2)
        print(delta_h1)        
        print("\n--------------------------\n")

        ## update rule
        # Hidden layer 1
        #(3,4)          (3,4)                      (3,1)       (1,4)
        h1_weights  = h1_weights  - (eta * np.dot(delta_h1[:,:3].T , np.reshape(x,((1,4))))) 

        # Hidden layer 2
        #(4,4)        (4,4)                       (4,1)             (1,4)
        h2_weights  = h2_weights - (eta * np.dot( delta_h2[:,:4].T , v1))
        
        # Output layer 
        #(2,5)        (2,5)                        (2,1)     (1,5)
        out_weights = out_weights - (eta * np.dot( delta_ow.T , v2))

        print("\n\nNew output weights: "+str(i)+"\n\n")
        print(out_weights)
        print("\n--------------------------\n")
    

print("\n\nHidden Layer 1 Weights:\n\n",h1_weights)

print("\n\nHidden Layer 2 Weights:\n\n",h2_weights)

print("\n\nOutput Weights:\n\n",out_weights)

# plot it
if(do_plot):   
    plt.plot(err)
    plt.ylabel('error')
    plt.xlabel('epochs')
    plt.show()