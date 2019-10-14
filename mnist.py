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


## data set
train_dim    = 2000
test_dim     = 2000
train_data   = np.loadtxt('./data/mnist_train.csv',delimiter=',',max_rows=train_dim)
test_data    = np.loadtxt('./data/mnist_test.csv',delimiter=',',max_rows=test_dim)

train_labels = train_data[:,0]
train_data   = train_data[:,1:]
train_data   = np.reshape(train_data,(train_dim,28,28))
train_data   = train_data[:,::2,::2] / 255.0
#train_data   = train_data[:,::2,::2] * (.99 / 255.0) + .01 ??
train_data   = np.reshape(train_data,(train_dim,196))

test_labels  = test_data[:,0]
test_data    = test_data[:,1:]
test_data    = np.reshape(test_data,(test_dim,28,28))
test_data    = test_data[:,::2,::2] / 255.0
test_data    = np.reshape(test_data,(test_dim,196))

#Fix labels to one-hot encoding
tl = np.zeros((train_dim,10))
for i,l in enumerate(train_labels):
    tl[i][int(l)] = 1
train_labels = tl

tl = np.zeros((test_dim,10))
for i,l in enumerate(test_labels):
    tl[i][int(l)] = 1
test_labels = tl

print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)
print(train_labels)


# weights with random numbers
h1_weights  = np.random.uniform(low=.1, high=1.0, size=(100,197))
out_weights = np.random.uniform(low=.1, high=1.0, size=(10,101))

do_plot     = True
eta         = .0001    # learning rate
max_epoch   = 1     # how many epochs? (each epoch will run through all 4 data points)
err         = np.zeros((max_epoch,1))   # lets record error to plot (get a convergence plot)
v1          = np.ones((1,101))
o           = np.ones((1,10))

# init error
err = np.zeros((max_epoch,len(train_data)))    

###############################################
# Training
###############################################
for k in range(max_epoch): 
       

    for i,x in enumerate(train_data):
        ## forward pass

        #(1,100)           (1,4)      (4,3)    
        for j in range(100):
            # layer 1 -- v1 is (1,4), x is (1,4), h1_w[j] is (1,4) 
            v1[0][j] = np.dot(np.append(x,1), np.transpose(h1_weights[j]))
            v1[0][j] = tanh(v1[0][j])

        #(1,10)          (1,5)      (5,2)    
        for j in range(10):
            # output layer -- o is (1,2), v2 is (1,5), out_w[j] is (1,5)
            o[0][j] = np.dot(v1, np.transpose(out_weights[j]))
            o[0][j] = o[0][j]
        

        ## error
        # y is (8,2), o is (2,1)
        err[k][i] = np.sum(((1.0/2.0) * np.power((o.T - train_labels[i]), 2.0)))


        ## backprop
        # Output layer
        #(10,1)                  (10,1)             (10,1) 
        delta_ow     = (-1.0) * (np.array(train_labels[i]) - np.reshape(o,((10,))))
         
        # Layer 1
        #(3,1)           (1,10)(w/o bias update) (10,100)           (1,100)
        delta_h1     = (np.dot(delta_ow.T, out_weights[:,:100]) * tanh(v1[:,:100],derive=True)).T


        ## update rule
        # Output layer 
        for j in range(10):
            out_weights[j] -= eta * v1.ravel() * delta_ow[j]
        
        # Hidden layer 1
        for j in range(100):
            h1_weights[j] -= eta * np.append(x,1) * delta_h1[j]
    

        print("\n\nHidden Layer 1 Weights: "+str(i)+"\n\n",h1_weights)

        print("\n\nOutput Weights:\n\n",out_weights)

# plot it
if(do_plot):   
    plt.plot(err.ravel())
    plt.ylabel('error')
    plt.xlabel('epochs')
    plt.show()