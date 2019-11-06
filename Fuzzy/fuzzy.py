import numpy as numpy
from matplotlib import pyplot as plt

class Fuzzy:
    def __init__(self,):
        print('init')
    
    def rule1(self,x):
        if(x < .25):
            return 'low'
        elif(x >= .25 and x <= .75):
            return 'medium'
        elif(x > .75):
            return 'high'



if(__name__ == '__main__'):
    fuz = Fuzzy()
    print(fuz.rule1(.3))