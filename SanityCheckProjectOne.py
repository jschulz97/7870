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


# some overall program parameters

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


# Lets run this algorithm already


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

        print("\n\nActivations: "+str(i)+"\n\n")
        print(v1)
        print(v2)
        print(o1,o2)
        print("\n--------------------------\n")
        
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
        delta_1_store = np.ones((3,1)) # local errors, which we will need to remember/store!
        delta_1_ow = np.ones((4, 3)) # so ( (weight index), (neuron index) )
        for n in range(3): # loop over neurons in this layer
            # back error, now its 4 error terms
            delta_1 = (delta_2_store[0]*n2_w1[n]) + (delta_2_store[1]*n2_w2[n]) + (delta_2_store[2]*n2_w3[n]) + (delta_2_store[3]*n2_w4[n])
            # this error
            local_nonlin = MyNonLinearity(v1[n],derive=True)
            delta_1_store[n] = delta_1 * local_nonlin
            # what is the delta update?
            for m in range(4):
                delta_1_ow[m,n] = X[inx,m] * (delta_1 * local_nonlin)

        print("\n\noutput deltas: "+str(i)+"\n\n")
        print(delta_3_1,delta_3_2)
        print(delta_2_store)
        print(delta_1_store)
        print("\n--------------------------\n")
                  
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

        print("\n\nNew output weights: "+str(i)+"\n\n")
        print(n3_w1,n3_w2)
        
        print("\n--------------------------\n")


# Final weights

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

# plot it        
plt.plot(err)
plt.ylabel('error')
#plt.show()


# What do we get now value wise?

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

