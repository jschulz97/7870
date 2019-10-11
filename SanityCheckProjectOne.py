#!/usr/bin/env python
# coding: utf-8
Hi class, here is what I got (you can run and verify below)

layer 1 ########################

n1_w1 (i.e., network layer 1, weight 1)
[[0.08860685]
 [0.20195729]
 [0.29157565]
 [0.19372863]]
 
n1_w2 (i.e., network layer 1, weight 2)
[[0.09033127]
 [0.1063255 ]
 [0.09675819]
 [0.10019741]]
 
n1_w3
[[0.29310792]
 [0.30129798]
 [0.29182387]
 [0.89285427]]
 
layer 2 ########################

n2_w1
[[0.0419728 ]
 [0.01799462]
 [0.05814251]
 [0.0664807 ]]
 
n2_w2
[[0.11446754]
 [0.10333934]
 [0.14187737]
 [0.25860662]]
 
n2_w3
[[0.1264505 ]
 [0.11094494]
 [0.14033488]
 [0.0477684 ]]
 
n2_w4
[[ 0.197662  ]
 [ 0.19885226]
 [ 0.19652186]
 [-0.10378586]]
 
layer 3 ########################

n3_w1
[[ 1.4551862 ]
 [ 1.20914277]
 [ 0.98307026]
 [ 0.01152043]
 [-0.15541029]]
 
n3_w2
[[ 0.01368568]
 [ 0.79565676]
 [ 0.10254178]
 [-0.00790254]
 [-0.08804369]]
# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as plticker
import ipywidgets as widgets
from IPython.display import display
from IPython.display import clear_output
import sys
import random


# Lets set up that tanh function

# In[ ]:


# I kept this in so you can use it if you really really really want to!
def sigmoid(x, derive=False):
    if derive: # the derivative 
        return x * (1.0 - x) # recall, you probably like to see it like this: sigmoid(x)(1-sigmoid(x)), I call it different than you maybe do
    z = np.exp(-x)
    return 1.0 / (1.0 + z) 

# make our tanh function 
def tanh(x, derive=False):
    if derive:
        return (1.0 - x*x)
    return ( (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)) )

# lets plot our tanh fx on -2 to 2 in increments of 0.1

test_neg_x = (-1.0) * np.arange(0.1,2.0,0.1) # negative numbers (start at 0.1, go to 2.0, in inc's of 0.1)
test_neg_x = np.flip( test_neg_x ) # put them in the order we need to make an array tomorrow
test_pos_x = np.arange(0,2.0,0.1) # now make the positive numbers
test_x = np.append(test_neg_x,test_pos_x) # put the arrays together
tanhres = tanh(test_x) # call our function on the array
plt.plot(tanhres) # plot it!

# now, look at what its deriv looks like

tanhderiv = tanh(tanhres,derive=True)
plt.plot(tanhderiv,'--r')
plt.ylabel('tanh and its der')
plt.legend(['tanh','tanh der'])
plt.show()


# some overall program parameters

# In[ ]:


# what function do we want to use? (tanh or sigmoid? point it to the function you want)
MyNonLinearity = tanh # sigmoid

# include nonlinearity on our output layer? (in the handout I said do not do)
IncludeNonLinOnOutput = 0 # 1

# learning rate
eta = 0.1

# how many epochs? (each epoch will pass all 8 data points through)
# so, epoch = 1 is one pass of the 8 data points
epoch = 1

# do we want to randomly select our samples at each epoch?
# in the handout, I told you to NOT do
RandomlySamplePoints = 0


# make our data set

# In[ ]:


# define the data set
X = np.array([
    [0, 0, 0, 1],  # data point (x,y,z) with homogenization (so last element is 1 for bias)
    [0, 0, 1, 1],  
    [0, 1, 0, 1],
    [0, 1, 1, 1],
    [1, 0, 0, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 1, 1, 1],
]) 
# its labels
y = np.array([[1, 0], # label, so (output 1 value, output 2 value)
              [0, 1], 
              [0, 1],
              [1, 0],
              [0, 1],
              [1, 0],
              [1, 0],
              [1, 0],
             ])


# init weights

# In[ ]:


RandomInitOrPreSeed = 2 # 1 means do random 
                        # 2 means use what I put online
                        # 3 means all zeros (try it, what do you expect...?)
if(RandomInitOrPreSeed==1): # random
    # layer 1
    n1_w1 = np.random.normal(0,1,(4, 1))
    n1_w2 = np.random.normal(0,1,(4, 1))
    n1_w3 = np.random.normal(0,1,(4, 1))
    # layer 2
    n2_w1 = np.random.normal(0,1,(4, 1))
    n2_w2 = np.random.normal(0,1,(4, 1))
    n2_w3 = np.random.normal(0,1,(4, 1))
    n2_w4 = np.random.normal(0,1,(4, 1))
    # output layer
    n3_w1 = np.random.normal(0,1,(5, 1))
    n3_w2 = np.random.normal(0,1,(5, 1))
elif(RandomInitOrPreSeed==2): # what I put online
    # layer 1
    n1_w1 = np.reshape(np.asarray([0.1, 0.2, 0.3, 0.2]),[4,1])
    n1_w2 = np.reshape(np.asarray([0.1, 0.1, 0.1, 0.1]),[4,1])
    n1_w3 = np.reshape(np.asarray([0.3, 0.3, 0.3, 0.9]),[4,1])
    # layer 2
    n2_w1 = np.reshape(np.asarray([0, 0, 0, 0]),[4,1])
    n2_w2 = np.reshape(np.asarray([0.1, 0.1, 0.1, 0.2]),[4,1])
    n2_w3 = np.reshape(np.asarray([0.1, 0.1, 0.1, 0.0]),[4,1])
    n2_w4 = np.reshape(np.asarray([0.2, 0.2, 0.2, -0.1]),[4,1])
    # output layer
    n3_w1 = np.reshape(np.asarray([1.5, 1.2, 1, 0, -0.2]),[5,1])
    n3_w2 = np.reshape(np.asarray([0, 0.8, 0.1, 0, -0.1]),[5,1]) 
elif(RandomInitOrPreSeed==3): # all 0's
    # layer 1
    n1_w1 = np.reshape(np.asarray([0, 0, 0, 0]),[4,1])
    n1_w2 = np.reshape(np.asarray([0, 0, 0, 0]),[4,1])
    n1_w3 = np.reshape(np.asarray([0, 0, 0, 0]),[4,1])
    # layer 2
    n2_w1 = np.reshape(np.asarray([0, 0, 0, 0]),[4,1])
    n2_w2 = np.reshape(np.asarray([0, 0, 0, 0]),[4,1])
    n2_w3 = np.reshape(np.asarray([0, 0, 0, 0]),[4,1])
    n2_w4 = np.reshape(np.asarray([0, 0, 0, 0]),[4,1])
    # output layer
    n3_w1 = np.reshape(np.asarray([0, 0, 0, 0, 0]),[5,1])
    n3_w2 = np.reshape(np.asarray([0, 0, 0, 0, 0]),[5,1]) 
    
print("layer 1 ########################")

print("n1_w1")
print(n1_w1)
print("n1_w2")
print(n1_w2)
print("n1_w3")
print(n1_w3)

print("layer 2 ########################")

print("n2_w1")
print(n2_w1)
print("n2_w2")
print(n2_w2)
print("n2_w3")
print(n2_w3)
print("n2_w4")
print(n2_w4)

print("layer 3 ########################")

print("n3_w1")
print(n3_w1)
print("n3_w2")
print(n3_w2)


# Lets run this algorithm already!

# In[ ]:


err = np.zeros((epoch,1)) # lets record error to plot (get a convergence plot)
inds = np.asarray([0,1,2,3,4,5,6,7]) # array of our 8 indices (data point index references)
for k in range(epoch): 
    
    # init error to zero per epoch
    err[k] = 0    
    
    # random shuffle data at each epoch?
    if(RandomlySamplePoints==1):
        inds = np.random.permutation(inds)
    
    # go through our data points
    for i in range(8): 
        
        # what index?
        inx = inds[i]
        
        # forward pass #########################
        
        # layer 1 ##############################
        v1 = np.ones((4, 1))
        v1[0] = np.dot(X[inx,:], n1_w1)            # neuron 1 fires (x as input)
        v1[0] = MyNonLinearity(v1[0])              # neuron 1 nonlinearity
        v1[1] = np.dot(X[inx,:], n1_w2) 
        v1[1] = MyNonLinearity(v1[1])  
        v1[2] = np.dot(X[inx,:], n1_w3) 
        v1[2] = MyNonLinearity(v1[2])          
        # layer 2 ##############################
        v2 = np.ones((5, 1))
        v2[0] = np.dot(np.transpose(v1), n2_w1) 
        v2[0] = MyNonLinearity(v2[0])             
        v2[1] = np.dot(np.transpose(v1), n2_w2)
        v2[1] = MyNonLinearity(v2[1])  
        v2[2] = np.dot(np.transpose(v1), n2_w3) 
        v2[2] = MyNonLinearity(v2[2]) 
        v2[3] = np.dot(np.transpose(v1), n2_w4)
        v2[3] = MyNonLinearity(v2[3])         
        # output layer #########################
        o1 = np.dot(np.transpose(v2), n3_w1) 
        o2 = np.dot(np.transpose(v2), n3_w2)         
        if(IncludeNonLinOnOutput==1): # do we run a nonlinearity?...
            o1 = MyNonLinearity(oo1) 
            o2 = MyNonLinearity(oo2) 
        
        # overall error ########################
        
        err[k] = err[k] + (1.0/2.0) * ( np.power((o1 - y[inx,0]), 2.0) 
                                       + np.power((o2 - y[inx,1]), 2.0) )
                
        # backprop time!!! #####################
                
        # output layer #########################
        delta_3_1 = (-1.0) * (y[inx,0] - o1)  
        delta_3_2 = (-1.0) * (y[inx,1] - o2)  
        if(IncludeNonLinOnOutput==1):
            delta_3_1 = delta_3_1 * MyNonLinearity(o1,derive=True)
            delta_3_2 = delta_3_2 * MyNonLinearity(o2,derive=True)
        # what should the update be?
        delta_3_1_ow = np.ones((5, 1))
        for m in range(5):
            delta_3_1_ow[m] = v2[m] * delta_3_1
        delta_3_2_ow = np.ones((5, 1))
        for m in range(5):
            delta_3_2_ow[m] = v2[m] * delta_3_2
            
        # hidden layer 2 #######################
        delta_2_store = np.ones((4,1)) # local errors, which we will need to remember/store!
        delta_2_ow = np.ones((4, 4)) # so ( (weight index), (neuron index) )
        for n in range(4): # loop over neurons in this layer
            # back error, so weighted version of the errors on the two neurons "down stream from us"
            delta_2 = (delta_3_1 * n3_w1[n]) + (delta_3_2 * n3_w2[n])
            # this error
            local_nonlin = MyNonLinearity(v2[n],derive=True)
            # remember its error for later
            delta_2_store[n] = delta_2 * local_nonlin
            # what is the delta update?
            for m in range(4):
                delta_2_ow[m,n] = v1[m] * delta_2_store[n]

        # hidden layer 1 #######################
        delta_1_ow = np.ones((4, 3)) # so ( (weight index), (neuron index) )
        for n in range(3): # loop over neurons in this layer
            # back error, now its 4 error terms
            delta_1 = (delta_2_store[0]*n2_w1[n]) + (delta_2_store[1]*n2_w2[n]) + (delta_2_store[2]*n2_w3[n]) + (delta_2_store[3]*n2_w4[n])
            # this error
            local_nonlin = MyNonLinearity(v1[n],derive=True)
            # what is the delta update?
            for m in range(4):
                delta_1_ow[m,n] = X[inx,m] * (delta_1 * local_nonlin)
                  
        # lets now update!!! ###################
                        
        # output layer
        n3_w1 = n3_w1 + ((-1)*eta) * delta_3_1_ow
        n3_w2 = n3_w2 + ((-1)*eta) * delta_3_2_ow        
        # hidden layer 2
        n2_w1 = n2_w1 + ((-1)*eta) * np.reshape(delta_2_ow[:,0],[4,1])
        n2_w2 = n2_w2 + ((-1)*eta) * np.reshape(delta_2_ow[:,1],[4,1])
        n2_w3 = n2_w3 + ((-1)*eta) * np.reshape(delta_2_ow[:,2],[4,1])
        n2_w4 = n2_w4 + ((-1)*eta) * np.reshape(delta_2_ow[:,3],[4,1])
        # hidden layer 1
        n1_w1 = n1_w1 + ((-1)*eta) * np.reshape(delta_1_ow[:,0],[4,1])
        n1_w2 = n1_w2 + ((-1)*eta) * np.reshape(delta_1_ow[:,1],[4,1])
        n1_w3 = n1_w3 + ((-1)*eta) * np.reshape(delta_1_ow[:,2],[4,1])


# Final weights

# In[ ]:


print("layer 1 ########################")

print("n1_w1")
print(n1_w1)
print("n1_w2")
print(n1_w2)
print("n1_w3")
print(n1_w3)

print("layer 2 ########################")

print("n2_w1")
print(n2_w1)
print("n2_w2")
print(n2_w2)
print("n2_w3")
print(n2_w3)
print("n2_w4")
print(n2_w4)

print("layer 3 ########################")

print("n3_w1")
print(n3_w1)
print("n3_w2")
print(n3_w2)


# Show results

# In[ ]:


# plot it        
plt.plot(err)
plt.ylabel('error')
plt.show()


# What do we get now value wise?

# In[ ]:


# what were the values (just do forward pass)  
for i in range(8):  
    
    # forward pass
    # layer 1
    v1 = np.ones((4, 1))
    v1[0] = np.dot(X[i,:], n1_w1) 
    v1[0] = MyNonLinearity(v1[0]) 
    v1[1] = np.dot(X[i,:], n1_w2) 
    v1[1] = MyNonLinearity(v1[1])  
    v1[2] = np.dot(X[i,:], n1_w3) 
    v1[2] = MyNonLinearity(v1[2])          
    # layer 2
    v2 = np.ones((5, 1))
    v2[0] = np.dot(np.transpose(v1), n2_w1) 
    v2[0] = MyNonLinearity(v2[0])             
    v2[1] = np.dot(np.transpose(v1), n2_w2)
    v2[1] = MyNonLinearity(v2[1])  
    v2[2] = np.dot(np.transpose(v1), n2_w3) 
    v2[2] = MyNonLinearity(v2[2]) 
    v2[3] = np.dot(np.transpose(v1), n2_w4)
    v2[3] = MyNonLinearity(v2[3])         
    # output layer
    o1 = np.dot(np.transpose(v2), n3_w1) 
    o2 = np.dot(np.transpose(v2), n3_w2)         
    if(IncludeNonLinOnOutput==1):
        o1 = MyNonLinearity(oo1) 
        o2 = MyNonLinearity(oo2) 
    
    print( str(i) + ": produced: " + str(o1) + " " + str(o2) + " wanted " + str(y[i,0]) + " " + str(y[i,1]) )

