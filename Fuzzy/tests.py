import numpy as np
from matplotlib import pyplot as plt
from definitions import *
from fuzzy import *

mfs = [ R_Shaped(2,3),
        L_Shaped(7,9),
        Triangle(4,5,7),
        S_Shaped(6,9),
        Z_Shaped(1,4),
        Trapezoid(2,3,7,9),
        Gaussian(5,.5),
        ]

def test_mf():
    #testx = [0,1,2,3,4,5,6,7,8,9]
    testx = [i/10 for i in range(0,100)]

    for mf in mfs:
        y = [mf.compute(x) for x in testx]
        plt.plot(testx,y)
        plt.show()    



def prob_7_3():
    a_char1 = {1: 1.0, 2: .4, 3: .1}
    b_char1 = {'a': 0.2, 'b': .8}
    c_char1 = {'w': 0.0, 'x': .4, 'y': .8, 'z': 1.0}

    fuz = Fuzzy_Model()
    
    fuz.add_input('A')
    fuz.add_fs_in('A','char1',a_char1)
    fuz.add_input('B')
    fuz.add_fs_in('B','char1',b_char1)

    fuz.add_output('C')
    fuz.add_fs_out('C','char1',c_char1)

    r = Rule(['a','char1'],['w','char1'])
    fuz.add_rule(r)

    print('U is A')
    fuz.build_zadeh_mat(np.array([1,.4,.1]))
    print('U is NOT A')
    fuz.build_zadeh_mat(np.array([.1,.4,1]))
    print('U is A`')
    fuz.build_zadeh_mat(np.array([.6,1,0]))


def tipping_problem_scitkit():
    fuz  = Fuzzy_Model(Zadeh())

    
