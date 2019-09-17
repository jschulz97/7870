# libs
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from IPython.display import clear_output
import sys
import random
import torch.nn as nn

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

model = nn.Sigmoid()

model()