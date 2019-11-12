import numpy as np

#Improved dictionary auto-adds keys when assigned and doesn't exist
class dict2(dict):
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


# S-Shaped membership function
def smem(x, a, b):
    if(x <= a):
        return 0
    elif(x > a and x <= (a+b)/2):
        return 2 * pow(((x-a)/(b-a)) , 2)
    elif(x > (a+b)/2 and x <= b):
        return 1 - (2 * pow(((x-b)/(b-a)) , 2))
    elif(x > b):
        return 1


# Z-Shaped membership function
def zmem(x, a, b):
    if(x <= a):
        return 1
    elif(x > a and x <= (a+b)/2):
        return 1 - (2 * pow(((x-a)/(b-a)) , 2))
    elif(x > (a+b)/2 and x <= b):
        return 2 * pow(((x-b)/(b-a)) , 2)
    elif(x > b):
        return 0


def zadeh(a,b):
    return min(1,(1-a+b))

#def trap_mem()


# Perform correlational product
# Input x,y as dictionaries
# Output nested dict
def corr_prod(x, y):
    d2 = dict2()

    for i in x.keys():
        for j in y.keys():
            d2[i][j] =  x[i] * y[j]

    return d2


# Perform correlational minimum
# input x,y as dictionaries
# Output nested dict
def corr_min(x, y):
    d2 = dict2()

    for i in x.keys():
        for j in y.keys():
            d2[i][j] =  min(x[i] , y[j])

    return d2
