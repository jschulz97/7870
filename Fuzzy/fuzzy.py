import numpy as np
from matplotlib import pyplot as plt
from definitions import *
import math

plt.rcParams.update({'font.size': 18})

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

    def add_fuzzy_set(self, name, val, mfx,):
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

    def aggregate_outputs(self, var,):
        #Implication/Discretization
        testx = self.consequents[var]['domain']
        Y = np.zeros((len(self.out_fxs),len(testx)))
        for i,fx in enumerate(self.out_fxs):
            y = fx.max
            fx.max = 1
            for j,x in enumerate(testx):
                Y[i,j] = self.op(y,fx.compute(x))

        #Aggregate output memberships
        self.By = np.zeros(len(testx))
        self.Bx = testx
        for y in Y:    
            for i,x in enumerate(testx):    
                self.By[i] += y[i]
        
        #Bring back down to 1 if greater
        self.By = [min(1,y) for y in self.By]   

    def fuzzy_out(self, var, y):
        fuz_mem = 0 
        fuz_val = ''
        for val in self.consequents[var]:
            if(val != 'domain'):
                obj = self.consequents[var]
                if(obj[val]['mfx'].compute(y) > fuz_mem):
                    fuz_val = val
        
        return fuz_val

    # Bx is x for output discrete set
    # By is aggregation of all rule firings
    def df_centroid(self,):
        sum0 = 0.0
        sum1 = 0.0

        for x,y in zip(self.Bx, self.By):
            sum0 += x * y
            sum1 += y

        return sum0/sum1

    def view_sets(self, var,):
        if(var in self.antecedents):
            obj = self.antecedents
        elif(var in self.consequents):
            obj = self.consequents
        else:
            print('Unknown Variable:',var)
            return
        
        for st in obj[var]:
            xaxis = obj[var]['domain']
            if(st != 'domain'):
                obj1 = obj[var][st]
                y = [obj1['mfx'].compute(x) for x in xaxis]
                plt.plot(xaxis,y)
        plt.ylim(0,1.05)
        plt.title(var+' Fuzzy Sets')
        plt.ylabel('membership')
        plt.xlabel('difference')
        plt.show()   

    def plot(self, title, xl, yl):
        #Plot!
        fig = plt.figure()
        fig.set_size_inches(8, 7, forward=True)
        plt.plot(self.Bx,self.By)
        plt.ylim(0,1.05)
        plt.xlim(0,1.0)
        plt.title(title)
        plt.ylabel(yl)
        plt.xlabel(xl)        
        plt.show()   