import numpy as numpy
from matplotlib import pyplot as plt
from mem_fxs import *

def test_mf():
    testx = [0,1,2,3,4,5,6,7,8,9]
    testy = [zmem(i,1,8) for i in testx]
    plt.plot(testx,testy)
    plt.show()



