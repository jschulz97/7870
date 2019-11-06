import numpy as numpy
from matplotlib import pyplot as plt
from mem_fxs import *

X = {1: 1.0, 2: .8, 3: 0.0, 4: 0.0}
Y = {'a': 0.0, 'b': .5, 'c': 1.0, 'd': .5, 'e': 0.0}

class Rule:
    def __init__(self, A, P, C, ):
        self.A = A
        self.P = P 
        self.C = C 


class Fuzzy_Model:
    def __init__(self, ):
        print('init')
        self.Rules = []
    

    def add_rule(self, rule_temp):
        self.Rules.append(rule_temp)


    def df_centroid(self, y, B):
        sum0 = 0.0
        sum1 = 0.0

        for yi in y:
            sum0 += yi * B[yi]
            sum1 += B[yi]

        return sum0/sum1


if(__name__ == '__main__'):
    fuz = Fuzzy_Model()
    #fuz.add_rule(Rule())
    