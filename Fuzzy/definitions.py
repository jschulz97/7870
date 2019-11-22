import numpy as np
from matplotlib import pyplot as plt
import math

##############################
# Rule Class
class Rule:
    def __init__(self, A, C, ):
        next_op = None
        self.A = []
        for a in A:
            if(type(a) == str):
                next_op = a
            else:
                self.A.append(dict2())
                self.A[-1]['op']   = next_op
                self.A[-1]['name'] = a['name']
                self.A[-1]['mfx']  = a['mfx']
        
        self.C = C


##############################
# Operator Class
# Base class for all operators
class Operator:
    def __init__(self,):
        _=0

    def __call__(self, a, b,):
        print('Base Operator Class is not callable')
        return 0

class Zadeh(Operator):
    def __call__(self, a, b,):
        return min(1, (1 - a + b))

class Corr_Prod(Operator):
    def __call__(self, a, b,):
        return (a * b)

class Corr_Min(Operator):
    def __call__(self, a, b,):
        return min(a, b)


##############################
# Mem function Class
# Base class for all mem fxs
class Membership_Function:
    def __init__(self, a, b, c=None, d=None,):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.max = 1
    
    def compute(self, x,):
        print('Base Membership Function Class is not computable')
        return 0

    def output(self,val):
        return min(val,self.max)

    def __repr__(self,):
        return str(type(self))

class R_Shaped(Membership_Function):
    def compute(self, x,):
        if(x > self.b):
            return self.output(0)
        elif(x >= self.a and x <= self.b):
            return self.output(((self.b - x)/(self.b - self.a)))
        elif(x < self.a):
            return self.output(1)

class L_Shaped(Membership_Function):
    def compute(self, x,):
        if(x > self.b):
            return self.output(1)
        elif(x >= self.a and x <= self.b):
            return self.output(((x - self.a)/(self.b - self.a)))
        elif(x < self.a):
            return self.output(0)

class Triangle(Membership_Function):
    def compute(self, x,):
        if(self.c == None):
            print('Triangle Membership Function needs 3 arguments on initialization')
            return 0
        
        if(x <= self.a):
            return self.output(0)
        elif(x > self.a and x <= self.b):
            return self.output(((x - self.a)/(self.b - self.a)))
        elif(x > self.b and x <= self.c):
            return self.output(((self.c - x)/(self.c - self.b)))
        elif(x >= self.c):
            return self.output(0)

class S_Shaped(Membership_Function):
    def compute(self, x,):
        if(x <= self.a):
            return self.output(0)
        elif(x > self.a and x <= (self.a + self.b)/2):
            return self.output(2 * pow(((x - self.a)/(self.b - self.a)) , 2))
        elif(x > (self.a + self.b)/2 and x <= self.b):
            return self.output(1 - (2 * pow(((x - self.b)/(self.b - self.a)) , 2)))
        elif(x > self.b): 
            return self.output(1)

class Z_Shaped(Membership_Function):
    def compute(self, x,):
        if(x <= self.a):
            return self.output(1)
        elif(x > self.a and x <= (self.a + self.b)/2):
            return self.output(1 - (2 * pow(((x - self.a)/(self.b - self.a)) , 2)))
        elif(x > (self.a + self.b)/2 and x <= self.b):
            return self.output(2 * pow(((x - self.b)/(self.b - self.a)) , 2))
        elif(x > self.b):
            return self.output(0)

class Trapezoid(Membership_Function):
    def compute(self, x,):
        if(self.c == None or self.d == None):
            print('Trapezoid Membership Function needs 4 arguments on initialization')
            return 0
        
        if(x < self.a):
            return self.output(0)
        elif(x >= self.a and x <= self.b):
            return self.output(((x - self.a)/(self.b - self.a)))
        elif(x >= self.b and x <= self.c):
            return self.output(1)
        elif(x >= self.c and x <= self.d):
            return self.output(((self.d - x)/(self.d - self.c)))
        elif(x > self.d):
            return self.output(0)

class Gaussian(Membership_Function):
    def compute(self, x,):
        return self.output(math.exp(-(pow(x-self.a,2)/(2 * pow(self.b,2)))))


def do_view(mf, a, b, step):
    testx = [i*step for i in range(int(a/step),int(b/step))]
    y = [mf['mfx'].compute(x) for x in testx]
    plt.plot(testx,y)
    plt.ylim(0,1.05)
    plt.ylabel('membership')
    plt.xlabel(mf['name'])
    plt.show()

    
#Improved dictionary auto-adds keys when assigned and doesn't exist
class dict2(dict):
    _=0
    # def __getitem__(self, item):
    #     try:
    #         return dict.__getitem__(self, item)
    #     except KeyError:
    #         print('dict2 key doesn\'t exist, creating new key:',item)
    #         value = self[item] = type(self)()
    #         return value

