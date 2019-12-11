# Let's do the above on a different data set now
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import time
from tqdm import tqdm
from IPython.display import display, clear_output
import math

from possi_flann_proj import *
from mpl_toolkits.axes_grid1 import AxesGrid
import progressbar
from generate_vat import *



def do_ga(it,mu,cr,disp=False):
    from init import X,Y,Z
    
    # how many iterations do we run?
    #NoIterations = 10
    NoIterations = it

    # what percentage of our chromosomes do we mutate?
    #MutationRate = 0.4
    MutationRate = mu

    # what is the crossover rate?
    #CrossoverRate = 0.5
    CrossoverRate = cr

    print('\nLoading Data...',end='')
    mon_train_features = np.load('./data/monkey_train.npy')
    mon_test_features  = np.load('./data/monkey_test.npy')
    tig_train_features = np.load('./data/tiger_train.npy')
    tig_test_features  = np.load('./data/tiger_test.npy')
    che_train_features = np.load('./data/cheetah_train.npy')
    che_test_features  = np.load('./data/cheetah_test.npy')
    print('Finished.')

    bound_factors = np.arange(1, 10, .1)
    training_nums = np.arange(10, 500, 5)
    test_dim = 100

    #Initialize a PKNN object
    PKNN   = Possi_FLANN(3,5,[tig_train_features,che_train_features],['tiger','cheetah'])
    score = PKNN.score([tig_test_features[:test_dim],che_test_features[:test_dim],mon_test_features[:test_dim]],['tiger','cheetah','none'],bound_factors[0],training_nums[0],False)
    #print('init score:',score)

    # make grid
    X, Y = np.meshgrid(X, Y)

    print(X.shape,Y.shape,Z.shape)

    plt.rcParams.update({'font.size': 20})
    
    # Plot the surface
    if(disp):
        fig = plt.figure( figsize=(15, 10) )
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cm.seismic, linewidth=0, antialiased=True, alpha=0.5)
        cset = ax.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.cool)
        ax.set_zlim(0, np.max(Z))
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.75, aspect=5)
        plt.show()
        plt.close()

    # generate our population
    Pop = 60
    Domain = np.array( [[1.0,10.0],[10,500]] )
    Chromos = np.zeros( (Pop,2) )
    for i in range(Pop):
        x = (np.random.rand() * (Domain[0,1] - Domain[0,0])) + Domain[0,0]
        y = (np.random.rand() * (Domain[1,1] - Domain[1,0])) + Domain[1,0]
        Chromos[i,0] = x
        Chromos[i,1] = y

    # boundary enforce
    Chromos[:,0] = np.maximum( Chromos[:,0], Domain[0,0] )
    Chromos[:,0] = np.minimum( Chromos[:,0], Domain[0,1] )
    Chromos[:,1] = np.maximum( Chromos[:,1], Domain[1,0] )
    Chromos[:,1] = np.minimum( Chromos[:,1], Domain[1,1] )    

    # figure setup
    if(disp):
        fig = plt.figure( figsize=(9, 7) )
        ax = fig.gca()
        cset = ax.contourf(X, Y, Z, cmap=cm.coolwarm, alpha=0.75)
        ax.plot(Chromos[:,0], Chromos[:,1], 'ok')
        plt.xlabel('Bound Factor')
        plt.ylabel('Num Fitting Images')
        plt.show()
        plt.close()

    CrossOverNumber = math.floor( Pop * CrossoverRate )

    # storage for the new chromosomes at each generation
    NewPopulation = np.zeros( (Pop,2) )
        
    # generations
    for i in range(NoIterations):
        
        # evaluate our chromsomes 
        Fitness = np.zeros(Pop)
        print('Scoring '+str(i)+'...',end='')
        for j in range(Pop):
            Fitness[j] = PKNN.score([tig_test_features[:test_dim],che_test_features[:test_dim],mon_test_features[:test_dim]],['tiger','cheetah','none'],Chromos[j,0],Chromos[j,1],False)
        print('Finished.')

        # elitism
        best = np.argmax( Fitness )
        NewPopulation[0,0] = Chromos[best,0]
        NewPopulation[0,1] = Chromos[best,1]    
            
        # select which to cross and cross
        Selections = np.zeros((Pop,2))
        for j in range(1,CrossOverNumber):
            # pick the selectors 
            Selections[j,0] = int(RouletteWheelSelection(Fitness))
            Selections[j,1] = int(RouletteWheelSelection(Fitness))
            if( Selections[j,0] == Selections[j,1] ):
                Selections[j,1] = Selections[j,1] % Pop
            # crossover
            NewPopulation[j,0] = Chromos[ int(Selections[j,0]) , 0 ]
            NewPopulation[j,1] = Chromos[ int(Selections[j,1]) , 1 ]
            
        # rest are selected parents    
        for j in range(CrossOverNumber,Pop):
            PickMe = int(RouletteWheelSelection(Fitness))
            NewPopulation[j,0] = Chromos[ int(PickMe) , 0 ]
            NewPopulation[j,1] = Chromos[ int(PickMe) , 1 ]
            
        # mutation
        for j in range(1,Pop):
            r = np.random.rand()
            if( r < MutationRate ):
                randomoffset = abs(Domain[0,0]) * np.asarray( [(np.random.rand() - 0.5), (np.random.rand() - 0.5)] )
                NewPopulation[j,:] = NewPopulation[j,:] + randomoffset
                
        # boundary enforce
        NewPopulation[:,0] = np.maximum( NewPopulation[:,0], Domain[0,0] )
        NewPopulation[:,0] = np.minimum( NewPopulation[:,0], Domain[0,1] )
        NewPopulation[:,1] = np.maximum( NewPopulation[:,1], Domain[1,0] )
        NewPopulation[:,1] = np.minimum( NewPopulation[:,1], Domain[1,1] )                                
                    
        # set new generation
        # Don't copy if last iteration
        if(i != NoIterations - 1):
            Chromos = np.copy(NewPopulation)

    print(max(Fitness))
    print(Fitness)
    print(Chromos[best,0])
    print(Chromos[best,1])
    PKNN.score([tig_test_features[:test_dim],che_test_features[:test_dim],mon_test_features[:test_dim]],['tiger','cheetah','none'],Chromos[best,0],Chromos[best,1],disp)

    # plot
    fig = plt.figure( figsize=(9, 7) )
    ax = fig.gca()
    ax.cla()
    cset = ax.contourf(X, Y, Z, cmap=cm.coolwarm, alpha=0.75)
    ax.plot(Chromos[:,0], Chromos[:,1], 'ok')
    ax.plot(Chromos[0,0], Chromos[0,1], 'xr', markersize=25)
    fig.suptitle('MR: '+str(MutationRate)+'  CR: '+str(CrossoverRate)+'\nBF: '+str(round(Chromos[best,0],2))+'  TN: '+str(round(Chromos[best,1],0))+'  Score: '+str(round(max(Fitness),2)))
    plt.xlabel('Bound Factor')
    plt.ylabel('Num Fitting Images')
    if(disp):
        plt.show()
    fig.savefig('./export/mu_'+str(MutationRate)+'_cr_'+str(CrossoverRate)+'.png')
    plt.close()


def RouletteWheelSelection( Fitnesses ):
    # what's its dimensionality?
    L = Fitnesses.size
    # for roulette wheel, flipping to "maximize" vs the reality that we are minimizing below
    #LocalFit = 1.0 / (Fitnesses + np.finfo(float).eps)
    LocalFit = Fitnesses
    LocalFit = LocalFit / np.sum(LocalFit)
    # lets do our picking
    i = 0
    # first check
    lsum = LocalFit[i]
    r = np.random.rand()
    while( (lsum < r) and (i < (L-1)) ): # consecutive checks
        i = i + 1
        lsum = lsum + LocalFit[i]
    return i