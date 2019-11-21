import numpy as np
from matplotlib import pyplot as plt
from definitions import *
from tests import *
import math

X = [1,2,3,4]
Y = ['a','b','c','d','e']
# small
A = {1: 1.0, 2: .8, 3: 0.0, 4: 0.0}
# medium
B = {'a': 0.0, 'b': .5, 'c': 1.0, 'd': .5, 'e': 0.0}



class Fuzzy_Model:
    def __init__(self,op=Operator(0,0)):
        self.op = op
        self.antecedents = dict2()
        # self.Rules = []
        # self.FS = dict2()
        # self.Inputs  = []
        # self.Outputs = []
        print('Init Fuzzy Model')

    def add_antecedent(self, name, domain):
        #Create dict2 for the antecedent
        self.antecedents[name] = dict2()
        #Assign domain to antecedent
        self.antecedents[name]['domain'] = domain
        #Initialize empty list of fuzzy values
        self.antecedents[name]['values'] = []
        print('Added Antecedent',name)

    def add_fuzzy_value(self, name, val, mfx,):
        #Create dict2 for fuzzy value and its mem fx
        d2 = dict2()
        d2[val] = mfx
        #Append it to the values list in the antecedent dict
        if(name not in self.antecedents):
            print('Antecedent name not valid')
        else:    
            self.antecedents[name]['values'].append(d2)
            print('Added fuzzy value',val,'to',name)

    def add_rule(self, rule)

    # def add_input(self,inp):
    #     self.FS[inp] = dict2()
    #     self.Inputs.append(inp)
    
    # def add_fs_in(self,inp,tag,fs):
    #     self.FS[inp][tag] = fs
    
    # def add_output(self,inp):
    #     self.FS[inp] = dict2()
    #     self.Outputs.append(inp)
    
    # def add_fs_out(self,inp,tag,fs):
    #     self.FS[inp][tag] = fs

    # def add_rule(self, rule_temp):
    #     self.Rules.append(rule_temp)

    
    # def fire(self, X):
    #     for j in range(U.shape[1]):
    #         for i in range(U.shape[0]):



    # y is set of outputs
    # B is aggregation of all rule firings
    def df_centroid(self, y, B):
        sum0 = 0.0
        sum1 = 0.0

        for yi in y:
            sum0 += yi * B[yi]
            sum1 += B[yi]

        return sum0/sum1

    # def build_zadeh_mat(self,inp):
    #     # print(self.FS)
    #     # print(self.FS['C']['char1'].keys())
    #     # print(self.FS['C']['char1'].values())
    #     arr = np.zeros((3,4))
    #     for i,a in enumerate(self.FS['A']['char1'].values()):
    #         for j,c in enumerate(self.FS['C']['char1'].values()):
    #             arr[i][j] = zadeh(a,c)
        
    #     print(arr)

    #     res = inp.T.dot(arr)
    #     print(res)


if(__name__ == '__main__'):
    #prob_7_3()
    test_mf()