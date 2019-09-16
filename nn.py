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
h1_n1_w  = [.1,.2,.3, .2]
h1_n2_w  = [.1,.1,.1, .1]
h1_n3_w  = [.3,.3,.3, .9]
h2_n1_w  = [ 0, 0, 0,  0]
h2_n2_w  = [.1,.1,.1, .2]
h2_n3_w  = [.1,.1,.1,  0]
h2_n4_w  = [.2,.2,.2,-.1]
out_n1_w = [1.5,1.2, 1,0,-.2]
out_n2_w = [  0, .8,.1,0,-.1]

# print('hidden layer 1, neuron 1 weights')
# print(n1_w)
# print('hidden layer 1, neuron 2 weights')
# print(n2_w)
# print('hidden layer 2, neuron 1 weights')
# print(n3_w)

###############################################
# Epochs
###############################################
eta = 3.0 # learning rate
err_break = 0.001 # stop when below this error
max_epoch = 20000 # how many epochs? (each epoch will run through all 4 data points)
err = np.zeros((max_epoch,1)) # lets record error to plot (get a convergence plot)
end_index = max_epoch-1 # what index did we stop on?
inds = np.asarray([0,1,2,3]) # array of our 4 indices (data point index references)
for k in range(max_epoch): 
    
    # init error
    err[k] = 0    
    
    # random shuffle of data each epoch?
    #inds = np.random.permutation(inds)
    
    # doing online, go through each point, one at a time
    for i in range(4): 
        
        # what index?
        inx = inds[i]
        
        # forward pass
        # layer 1
        v = np.ones((3, 1))
        v[0] = np.dot(X[inx,:], n1_w) # neuron 1 fires (x as input)
        v[0] = sigmoid(v[0])        # neuron 1 sigmoid
        v[1] = np.dot(X[inx,:], n2_w) # neuron 2 fires (x as input)
        v[1] = sigmoid(v[1])    
        # layer 2
        oo = np.dot(np.transpose(v), n3_w) # neuron 3 fires, taking neuron 1 and 2 as input
        o = sigmoid(oo) # hey, result of our net!!!
        
        # error
        err[k] = err[k] + ((1.0/2.0) * np.power((y[inx] - o), 2.0))
                
        # backprop time folks!!!
        
        # output layer, our delta is (delta_1 * delta_2)
        delta_1 = (-1.0) * (y[inx] - o)
        delta_2 = sigmoid(o,derive=True) # note how I called it, I passed o=sigmoid(oo)
        
        # now, lets prop it back to the weights
        delta_ow = np.ones((3, 1))
        # format is
        #  delta_index = (input to final neuron) * (Err derivative * Sigmoid derivative)
        delta_ow[0] = v[0]  *  (delta_1*delta_2)
        delta_ow[1] = v[1]  *  (delta_1*delta_2)
        delta_ow[2] = v[2]  *  (delta_1*delta_2)
        
        # neuron n1
        delta_3 = sigmoid(v[0],derive=True)
        # same, need to prop back to weights
        delta_hw1 = np.ones((3, 1))
        # format
        #              input     this Sig der     error from output   weight to output neuron
        delta_hw1[0] = X[inx,0]  *  delta_3  *  ((delta_1*delta_2)   *n3_w[0])
        delta_hw1[1] = X[inx,1]  *  delta_3  *  ((delta_1*delta_2)   *n3_w[0])
        delta_hw1[2] = X[inx,2]  *  delta_3  *  ((delta_1*delta_2)   *n3_w[0])     
        
        # neuron n2
        delta_4 = sigmoid(v[1],derive=True)
        # same, need to prop back to weights        
        delta_hw2 = np.ones((3, 1))
        delta_hw2[0] = X[inx,0]  *  delta_4  *   ((delta_1*delta_2)   *n3_w[1])
        delta_hw2[1] = X[inx,1]  *  delta_4  *   ((delta_1*delta_2)   *n3_w[1])
        delta_hw2[2] = X[inx,2]  *  delta_4  *   ((delta_1*delta_2)   *n3_w[1])
        
        # update rule
        n1_w = n1_w - eta * delta_hw1 # neuron 1 in hidden layer 1
        n2_w = n2_w - eta * delta_hw2 # neuron 2 in hidden layer 1
        n3_w = n3_w - eta * delta_ow  # neuron 1 in hidden layer 2
        
    if( err[k] < err_break ):
        end_index = k
        break

print('Ran ' + str(end_index) + ' iterations')

# plot it        
plt.plot(err[0:end_index])
plt.ylabel('error')
plt.xlabel('epochs')
plt.show()
        
# what were the values (just do forward pass)  
for i in range(4): 
    
    # forward pass
    v = np.ones((3, 1))
    v[0] = np.dot(X[i,:], n1_w)
    v[0] = sigmoid(v[0])
    v[1] = np.dot(X[i,:], n2_w)
    v[1] = sigmoid(v[1])    
    oo = np.dot(np.transpose(v), n3_w)
    o = sigmoid(oo) 
    print(str(i) + ": produced: " + str(o) + " wanted " + str(y[i]))