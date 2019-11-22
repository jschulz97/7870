import numpy as np
from matplotlib import pyplot as plt
from definitions import *
import math


class Fuzzy_Model:
    def __init__(self,op=None):
        self.op = op
        self.antecedents = dict2()
        self.consequents = dict2()
        self.rules = []
        self.inputs = dict2()
        print('Init Fuzzy Model')

    def add_antecedent(self, name, domain):
        #Create dict2 for the antecedent
        self.antecedents[name] = dict2()
        #Assign domain to antecedent
        self.antecedents[name]['domain'] = domain
        print('Added Antecedent:',name)

    def add_consequent(self, name, domain):
        #Create dict2 for the consequent
        self.consequents[name] = dict2()
        #Assign domain to consequent
        self.consequents[name]['domain'] = domain
        self.ydim = (domain[0],domain[-1])
        print('Added Consequent:',name)

    def add_fuzzy_value(self, name, val, mfx,):
        #Assign to correct name
        if(name in self.antecedents):
            self.antecedents[name][val] = dict2()
            self.antecedents[name][val]['mfx']  = mfx
            self.antecedents[name][val]['name'] = name
            print('Added fuzzy value',val,'to',name)
        elif(name in self.consequents):
            self.consequents[name][val] = dict2()
            self.consequents[name][val]['mfx']  = mfx
            self.consequents[name][val]['name'] = name
            print('Added fuzzy value',val,'to',name)
        else:
            print('Cannot add fuzzy value! Invalid Name:',name)

    def add_rule(self, rule,):
        self.rules.append(rule)

    def add_input(self, name, x,):
        if(name in self.antecedents):
            self.inputs[name] = x
        else:
            print('Cannot add input! Invalid fuzzy set:',name)

    # Let's get some outputs
    def fire(self,):
        print('\nFiring!\n')
        self.out_fxs = []
        for r in self.rules:
            val = 0
            #for each antecedent in the rule
            for a in r.A:
                #Get input variable for antecedent
                x = self.inputs[a['name']]
                #Get membership for input
                #If this isn't the first antecedent
                if(a['op'] is not None):
                    if(a['op'] == 'or'):
                        x_new = max(a['mfx'].compute(x),val)
                        val = x_new
                    elif(a['op'] == 'and'):
                        x_new = min(a['mfx'].compute(x),val)
                        val = x_new
                    else:
                        print('Undefined operation for multiple antecedents:',a['op'])
                else:
                    val = a['mfx'].compute(x)

            #Cap consequent activation
            r.C['mfx'].max = val
            self.out_fxs.append(r.C['mfx'])
            #View output for this rule
            #do_view(r.C,0,26,.1)

    def aggregate_outputs(self,samp_rate=.1):
        #Aggregate output memberships
        testx = [i*samp_rate for i in range(int(self.ydim[0]/samp_rate),int(self.ydim[1]/samp_rate))]
        self.By = np.zeros(len(testx))
        self.Bx = testx
        for fx in self.out_fxs:    
            for i,x in enumerate(testx):    
                self.By[i] += fx.compute(x)
        
        plt.plot(testx,self.By)
        plt.ylim(0,1)
        plt.ylabel('membership')
        plt.xlabel('output')
        plt.show()      


    # y is set of outputs
    # B is aggregation of all rule firings
    def df_centroid(self,):
        sum0 = 0.0
        sum1 = 0.0

        for x,y in zip(self.Bx, self.By):
            sum0 += x * y
            sum1 += y

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