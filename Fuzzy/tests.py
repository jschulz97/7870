import numpy as np
from matplotlib import pyplot as plt
from definitions import *
from fuzzy import *
import progressbar
import json
#import pprint

#pp = pprint.PrettyPrinter(indent=4)
models   = ['resnet18','resnet101','alexnet','google','vgg']
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


# Sanity Check
def tipping_problem_scikit():
    quality_domain = np.arange(0,11,1)
    service_domain = np.arange(0,11,1)
    tip_domain     = np.arange(0,26,1)

    quality_low    = Triangle(0,0,5)
    quality_med    = Triangle(0,5,10)
    quality_high   = Triangle(5,10,10)

    service_low    = Triangle(0,0,5)
    service_med    = Triangle(0,5,10)
    service_high   = Triangle(5,10,10)

    tip_low        = Triangle(0,0,13)
    tip_med        = Triangle(0,13,25)
    tip_high       = Triangle(13,25,25)

    fuz  = Fuzzy_Model(Corr_Min())

    fuz.add_antecedent('quality',quality_domain)
    fuz.add_antecedent('service',service_domain)
    fuz.add_consequent('tip',tip_domain)

    fuz.add_fuzzy_set('quality','poor',quality_low)
    fuz.add_fuzzy_set('quality','average',quality_med)
    fuz.add_fuzzy_set('quality','good',quality_high)

    fuz.add_fuzzy_set('service','poor',service_low)
    fuz.add_fuzzy_set('service','average',service_med)
    fuz.add_fuzzy_set('service','good',service_high)

    fuz.add_fuzzy_set('tip','low',tip_low)
    fuz.add_fuzzy_set('tip','med',tip_med)
    fuz.add_fuzzy_set('tip','high',tip_high)

    rule_1 = Rule( [fuz.antecedents['quality']['poor'], 'or', fuz.antecedents['service']['poor']],
                    fuz.consequents['tip']['low'])
    rule_2 = Rule( [fuz.antecedents['service']['average']],
                    fuz.consequents['tip']['med'])
    rule_3 = Rule( [fuz.antecedents['service']['good'], 'and', fuz.antecedents['quality']['good']],
                    fuz.consequents['tip']['high'])

    fuz.add_rule(rule_1)
    fuz.add_rule(rule_2)
    fuz.add_rule(rule_3)

    fuz.add_input('quality',6.5)
    fuz.add_input('service',9.8)

    fuz.fire()
    fuz.aggregate_outputs('Extendability')
    print('Fuzzy Answer:',fuz.fuzzy_out())
    print('Defuzzified Answer:',fuz.df_centroid())
    fuz.plot()


def get_distances():
    classes  = ['tiger','cheetah','monkey']
    models   = ['resnet18','resnet101','alexnet','google','vgg']
    tigers   = dict2()
    cheetahs = dict2()
    monkeys  = dict2()
    for md in models:
        tigers[md]   = np.load('./data/'+md+'_tigers.npy')[:100]
        cheetahs[md] = np.load('./data/'+md+'_cheetahs.npy')[:100]
        monkeys[md]  = np.load('./data/'+md+'_monkeys.npy')[:100]

    tiger_cheetah_dist  = dict2()
    cheetah_monkey_dist = dict2()
    tiger_monkey_dist   = dict2()

    tiger_tiger_dist    = dict2()
    cheetah_cheetah_dist= dict2()
    monkey_monkey_dist  = dict2()

    for md in progressbar.progressbar(models):
        tiger_cheetah_dist[md]   = []
        cheetah_monkey_dist[md]  = []
        tiger_monkey_dist[md]    = []
        tiger_tiger_dist[md]     = []
        cheetah_cheetah_dist[md] = []
        monkey_monkey_dist[md]   = []
        for i in progressbar.progressbar(range(0,100)):
            for j in range(0,100):
                if(i == j):
                    continue
                else:
                    tiger_cheetah_dist[md].append(np.linalg.norm(cheetahs[md][i] - tigers[md][j]))
                    cheetah_monkey_dist[md].append(np.linalg.norm(monkeys[md][i] - cheetahs[md][j]))
                    tiger_monkey_dist[md].append(np.linalg.norm(monkeys[md][i] - tigers[md][j]))
                    tiger_tiger_dist[md].append(np.linalg.norm(tigers[md][i] - tigers[md][j]))
                    cheetah_cheetah_dist[md].append(np.linalg.norm(cheetahs[md][i] - cheetahs[md][j]))
                    monkey_monkey_dist[md].append(np.linalg.norm(monkeys[md][i] - monkeys[md][j]))

    tc_stats = dict2()
    cm_stats = dict2()
    tm_stats = dict2()
    tt_stats = dict2()
    cc_stats = dict2()
    mm_stats = dict2()

    for md in models:
        tc_stats[md] = dict2()
        tc_stats[md]['u'] = str(np.mean(tiger_cheetah_dist[md]))
        tc_stats[md]['s'] = str(np.std(tiger_cheetah_dist[md]))
        cm_stats[md] = dict2()
        cm_stats[md]['u'] = str(np.mean(cheetah_monkey_dist[md]))
        cm_stats[md]['s'] = str(np.std(cheetah_monkey_dist[md]))
        tm_stats[md] = dict2()
        tm_stats[md]['u'] = str(np.mean(tiger_monkey_dist[md]))
        tm_stats[md]['s'] = str(np.std(tiger_monkey_dist[md]))
        tt_stats[md] = dict2()
        tt_stats[md]['u'] = str(np.mean(tiger_tiger_dist[md]))
        tt_stats[md]['s'] = str(np.std(tiger_tiger_dist[md]))
        cc_stats[md] = dict2()
        cc_stats[md]['u'] = str(np.mean(cheetah_cheetah_dist[md]))
        cc_stats[md]['s'] = str(np.std(cheetah_cheetah_dist[md]))
        mm_stats[md] = dict2()
        mm_stats[md]['u'] = str(np.mean(monkey_monkey_dist[md]))
        mm_stats[md]['s'] = str(np.std(monkey_monkey_dist[md]))
    
    stats        = [tc_stats,cm_stats,tm_stats,tt_stats,cc_stats,mm_stats]
    stats_labels = ['tc','cm','tm','tt','cc','mm']
    for st,lb in zip(stats,stats_labels):
        dump = json.dumps(st)
        with open('./data/'+lb+'.json',"w") as f:
            f.write(dump)      



def load_data():
    stats = dict2()
    stats_labels = ['tc','cm','tm','tt','cc','mm']
    
    for st in stats_labels:
        with open('./data/'+st+'.json',"r") as f:
            stats[st] = json.load(f)
    
    #pp.pprint(stats)
    return stats

def normalize(stats):
    sim_st  = ['tt','cc','mm']
    diff_st = ['tc','tm','cm']

    #Get max/min of each model's info
    u_range = dict2()
    s_range = dict2()
    for md in models:
        u_mx = float(stats['tc'][md]['u'])
        u_mi = float(stats['tc'][md]['u'])
        s_mx = float(stats['tc'][md]['s'])
        s_mi = float(stats['tc'][md]['s'])
        for st in stats:
            new_u = float(stats[st][md]['u'])
            new_s = float(stats[st][md]['s'])
            if(new_u > u_mx):
                u_mx = new_u
            if(new_u < u_mi):
                u_mi = new_u
            if(new_s > s_mx):
                s_mx = new_s
            if(new_s < s_mi):
                s_mi = new_s
        u_range[md] = (u_mi, u_mx)
        s_range[md] = (s_mi, s_mx)

    #pp.pprint(u_range)
    #pp.pprint(s_range)

    #normalize
    for md in models:
        for st in stats:
            stats[st][md]['u'] = float(stats[st][md]['u']) - u_range[md][0]
            stats[st][md]['u'] = float(stats[st][md]['u']) / (u_range[md][1] - u_range[md][0])
            stats[st][md]['s'] = float(stats[st][md]['s']) - s_range[md][0]
            stats[st][md]['s'] = float(stats[st][md]['s']) / (s_range[md][1] - s_range[md][0])
    
    return stats
    
def nplusone(stats):
    fuz = Fuzzy_Model(Corr_Min())
    fuz.add_antecedent('Nplusone_Distance',np.arange(0,1.01,.01))
    fuz.add_antecedent('Distance_to_Class1',np.arange(0,1.01,.01))
    fuz.add_antecedent('Distance_to_Class2',np.arange(0,1.01,.01))
    fuz.add_consequent('Extendability',np.arange(0,1.01,.01))

    fuz.add_fuzzy_set('Nplusone_Distance','low',Z_Shaped(.3,.6))
    fuz.add_fuzzy_set('Nplusone_Distance','med',Gaussian(.6,.15))
    fuz.add_fuzzy_set('Nplusone_Distance','high',S_Shaped(.6,.9))
    
    fuz.add_fuzzy_set('Distance_to_Class1','low',Z_Shaped(.3,.6))
    fuz.add_fuzzy_set('Distance_to_Class1','med',Gaussian(.6,.15))
    fuz.add_fuzzy_set('Distance_to_Class1','high',S_Shaped(.6,.9))

    fuz.add_fuzzy_set('Distance_to_Class2','low',Z_Shaped(.3,.6))
    fuz.add_fuzzy_set('Distance_to_Class2','med',Gaussian(.6,.15))
    fuz.add_fuzzy_set('Distance_to_Class2','high',S_Shaped(.6,.9))
    
    fuz.add_fuzzy_set('Extendability','low',R_Shaped(.3,.5))
    fuz.add_fuzzy_set('Extendability','med',Trapezoid(.2,.4,.8,.9))
    fuz.add_fuzzy_set('Extendability','high',L_Shaped(.7,.9))

    #fuz.view_sets('Class_1_Difference')
    #fuz.view_sets('Class_2_Difference')
    #fuz.view_sets('Difference_btn_Classes')
    #fuz.view_sets('Extendability')
    rules = []

    # More Explicit set of rules
    rules.append( Rule( [fuz.antecedents['Nplusone_Distance']['low'],'and',fuz.antecedents['Distance_to_Class1']['med'],'and',fuz.antecedents['Distance_to_Class2']['med']],
                         fuz.consequents['Extendability']['high']) )
    rules.append( Rule( [fuz.antecedents['Nplusone_Distance']['low'],'and',fuz.antecedents['Distance_to_Class1']['low'],'and',fuz.antecedents['Distance_to_Class2']['med']],
                         fuz.consequents['Extendability']['med']) )
    rules.append( Rule( [fuz.antecedents['Nplusone_Distance']['high'],'and',fuz.antecedents['Distance_to_Class1']['low'],'and',fuz.antecedents['Distance_to_Class2']['med']],
                         fuz.consequents['Extendability']['low']) )
    rules.append( Rule( [fuz.antecedents['Nplusone_Distance']['med'],'and',fuz.antecedents['Distance_to_Class1']['high'],'and',fuz.antecedents['Distance_to_Class2']['high']],
                         fuz.consequents['Extendability']['high']) )
  
    # General set of rules
    # rules.append( Rule( [fuz.antecedents['Distance_to_Class1']['low'],'or',fuz.antecedents['Distance_to_Class2']['low']],
    #                      fuz.consequents['Extendability']['low']) )
    # rules.append( Rule( [fuz.antecedents['Nplusone_Distance']['low']],
    #                      fuz.consequents['Extendability']['high']) )
    # rules.append( Rule( [fuz.antecedents['Nplusone_Distance']['low'],'and',fuz.antecedents['Distance_to_Class1']['high'],'and',fuz.antecedents['Distance_to_Class2']['high']],
    #                      fuz.consequents['Extendability']['high']) )
    # rules.append( Rule( [fuz.antecedents['Nplusone_Distance']['low'],'and',fuz.antecedents['Distance_to_Class1']['med'],'and',fuz.antecedents['Distance_to_Class2']['med']],
    #                      fuz.consequents['Extendability']['high']) )
    # rules.append( Rule( [fuz.antecedents['Nplusone_Distance']['med'],'and',fuz.antecedents['Distance_to_Class1']['med'],'and',fuz.antecedents['Distance_to_Class2']['med']],
    #                      fuz.consequents['Extendability']['med']) )

    for r in rules:
        fuz.add_rule(r)

    #Experimental Data from model outputs
    if(True):
        fuz_list = [fuz,fuz,fuz,fuz,fuz]
        for fuz,md in zip(fuz_list,models):
            d = stats['mm'][md]['u']
            c1 = stats['tm'][md]['u']
            c2 = stats['cm'][md]['u']
            print('x1:',d)
            print('x2:',c1)
            print('x3:',c2)
            fuz.add_input('Nplusone_Distance',d)
            fuz.add_input('Distance_to_Class1',c1)
            fuz.add_input('Distance_to_Class2',c2)

            fuz.fire()
            fuz.aggregate_outputs('Extendability')
            print('Fuzzy Answer:',fuz.fuzzy_out('Extendability',fuz.df_centroid()))
            print('Defuzzified Answer:',fuz.df_centroid())
            fuz.plot('N+1 Extendability of '+md+'\nInputs: '+str(round(d,2))+', '+str(round(c1,2))+', '+str(round(c2,2)),'Score: '+str(round(fuz.df_centroid(),4)),'Membership')
    
    #Hand crafted test values
    if(False):
        diff   = [.3,.2,.5,.7,.5]
        class1 = [.7,.2,.5,.3,1.0]
        class2 = [.7,.8,.5,.4,1.0]
        fuz_list = [fuz,fuz,fuz,fuz,fuz]
        for fuz,d,c1,c2 in zip(fuz_list,diff,class1,class2):
            print('x1:',d)
            print('x2:',c1)
            print('x3:',c2)
            fuz.add_input('Nplusone_Distance',d)
            fuz.add_input('Distance_to_Class1',c1)
            fuz.add_input('Distance_to_Class2',c2)

            fuz.fire()
            fuz.aggregate_outputs('Extendability')
            print('Fuzzy Answer:',fuz.fuzzy_out('Extendability',fuz.df_centroid()))
            print('Defuzzified Answer:',fuz.df_centroid())
            fuz.plot('N+1 Extendability\nInputs: '+str(d)+', '+str(c1)+', '+str(c2),'Score: '+str(round(fuz.df_centroid(),4)),'Membership')



if(__name__ == '__main__'):
    #test_mf()
    #tipping_problem_scikit()
    stats = load_data()
    stats = normalize(stats)
    nplusone(stats)








