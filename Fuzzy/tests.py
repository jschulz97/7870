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
    quality_domain = np.arange(0,11,1)
    service_domain = np.arange(0,11,1)
    tip_domain     = np.arange(0,26,1)

    quality_low    = Triangle(0,0,5)
    quality_med    = Triangle(0,5,10)
    quality_high   = Triangle(5,10,10)

    service_low    = Triangle(0,0,5)
    service_med    = Triangle(0,5,10)
    service_high   = Triangle(5,10,10)

    tip_low    = Triangle(0,0,13)
    tip_med    = Triangle(0,13,25)
    tip_high   = Triangle(13,25,25)

    fuz  = Fuzzy_Model(Zadeh())

    fuz.add_antecedent('quality',quality_domain)
    fuz.add_antecedent('service',service_domain)
    fuz.add_consequent('tip',tip_domain)

    fuz.add_fuzzy_value('quality','poor',quality_low)
    fuz.add_fuzzy_value('quality','average',quality_med)
    fuz.add_fuzzy_value('quality','good',quality_high)

    fuz.add_fuzzy_value('service','poor',service_low)
    fuz.add_fuzzy_value('service','average',service_med)
    fuz.add_fuzzy_value('service','good',service_high)

    fuz.add_fuzzy_value('tip','low',tip_low)
    fuz.add_fuzzy_value('tip','med',tip_med)
    fuz.add_fuzzy_value('tip','high',tip_high)

    rule_1 = Rule( [fuz.antecedents['quality']['poor'],
                    'or',
                    fuz.antecedents['service']['poor']],
                    fuz.consequents['tip']['low'])
    rule_2 = Rule( [fuz.antecedents['service']['average']],
                    fuz.consequents['tip']['med'])
    rule_3 = Rule( [fuz.antecedents['service']['good'],
                    'and',
                    fuz.antecedents['quality']['good']],
                    fuz.consequents['tip']['high'])

    fuz.add_rule(rule_1)
    fuz.add_rule(rule_2)
    fuz.add_rule(rule_3)

    fuz.add_input('quality',6.5)
    fuz.add_input('service',9.8)

    fuz.fire()
    fuz.aggregate_outputs()
    print(fuz.df_centroid())



if(__name__ == '__main__'):
    #prob_7_3()
    #test_mf()
    tipping_problem_scitkit()






