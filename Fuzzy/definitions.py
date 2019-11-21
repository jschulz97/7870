import numpy as np
import math

##############################
# Rule Class
class Rule:
    def __init__(self, A, C, ):
        # Dict {'set':[], 'value':[]}
        self.A = A

        self.C = C 

##############################
# Operator Class
# Base class for all operators
class Operator:
    def __init__(self,):
        _=0

    def fire(self, a, b,):
        print('Base Operator Class is not fireable')
        return 0

class Zadeh(Operator):
    def fire(self, a, b,):
        return min(1, (1- a + b))

class Corr_Prod(Operator):
    def fire(self, a, b,):
        return (a * b)

class Corr_Min(Operator):
    def fire(self, a, b,):
        return min(a, b)

# class Operator:
#     def __init__(self, a, b,):
#         self.a = a
#         self.b = b

#     def fire(self,):
#         print('Base Operator Class is not fireable')
#         return 0

# class Zadeh(Operator):
#     def fire(self,):
#         return min(1, (1- self.a + self.b))

# class Corr_Prod(Operator):
#     def fire(self,):
#         return (self.a * self.b)

# class Corr_Min(Operator):
#     def fire(self,):
#         return min(self.a, self.b)


##############################
# Mem function Class
# Base class for all mem fxs
class Membership_Function:
    def __init__(self, a, b, c=None, d=None,):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    
    def compute(self, x,):
        print('Base Membership Function Class is not computable')
        return 0

class R_Shaped(Membership_Function):
    def compute(self, x,):
        if(x > self.b):
            return 0
        elif(x >= self.a and x <= self.b):
            return ((self.b - x)/(self.b - self.a))
        elif(x < self.a):
            return 1

class L_Shaped(Membership_Function):
    def compute(self, x,):
        if(x > self.b):
            return 1
        elif(x >= self.a and x <= self.b):
            return ((x - self.a)/(self.b - self.a))
        elif(x < self.a):
            return 0

class Triangle(Membership_Function):
    def compute(self, x,):
        if(self.c == None):
            print('Triangle Membership Function needs 4 arguments on initialization')
            return 0
        
        if(x <= self.a):
            return 0
        elif(x > self.a and x <= self.b):
            return ((x - self.a)/(self.b - self.a))
        elif(x > self.b and x <= self.c):
            return ((self.c - x)/(self.c - self.b))
        elif(x >= self.c):
            return 0

class S_Shaped(Membership_Function):
    def compute(self, x,):
        if(x <= self.a):
            return 0
        elif(x > self.a and x <= (self.a + self.b)/2):
            return 2 * pow(((x - self.a)/(self.b - self.a)) , 2)
        elif(x > (self.a + self.b)/2 and x <= self.b):
            return 1 - (2 * pow(((x - self.b)/(self.b - self.a)) , 2))
        elif(x > self.b): 
            return 1

class Z_Shaped(Membership_Function):
    def compute(self, x,):
        if(x <= self.a):
            return 1
        elif(x > self.a and x <= (self.a + self.b)/2):
            return 1 - (2 * pow(((x - self.a)/(self.b - self.a)) , 2))
        elif(x > (self.a + self.b)/2 and x <= self.b):
            return 2 * pow(((x - self.b)/(self.b - self.a)) , 2)
        elif(x > self.b):
            return 0

class Trapezoid(Membership_Function):
    def compute(self, x,):
        if(self.c == None or self.d == None):
            print('Trapezoid Membership Function needs 4 arguments on initialization')
            return 0
        
        if(x < self.a):
            return 0
        elif(x >= self.a and x <= self.b):
            return ((x - self.a)/(self.b - self.a))
        elif(x >= self.b and x <= self.c):
            return 1
        elif(x >= self.c and x <= self.d):
            return ((self.d - x)/(self.d - self.c))
        elif(x > self.d):
            return 0

class Gaussian(Membership_Function):
    def compute(self, x,):
        return math.exp(-(pow(x-self.a,2)/(2 * pow(self.b,2))))


#Improved dictionary auto-adds keys when assigned and doesn't exist
class dict2(dict):
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


# Perform correlational product
# Input x,y as dictionaries
# Output nested dict
# def corr_prod(x, y):
#     d2 = dict2()

#     for i in x.keys():
#         for j in y.keys():
#             d2[i][j] =  x[i] * y[j]

#     return d2


# # Perform correlational minimum
# # input x,y as dictionaries
# # Output nested dict
# def corr_min(x, y):
#     d2 = dict2()

#     for i in x.keys():
#         for j in y.keys():
#             d2[i][j] =  min(x[i] , y[j])

#     return d2
