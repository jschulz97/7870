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

def RouletteWheelSelection( Fitnesses ):
    # what's its dimensionality?
    L = Fitnesses.size
    # for roulette wheel, flipping to "maximize" vs the reality that we are minimizing below
    LocalFit = 1.0 / (Fitnesses + np.finfo(float).eps)
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


# make grid
X = np.arange(-5, 5, 0.05)
Y = np.arange(-5, 5, 0.05)
X, Y = np.meshgrid(X, Y)
# new function!
A = 10
Z = A*2 + (X**2 - A*np.cos(2*math.pi*X)) + (Y**2 - A*np.cos(2*math.pi*Y))



# Plot the surface

fig = plt.figure( figsize=(15, 10) )
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.seismic, linewidth=0, antialiased=True, alpha=0.5)
cset = ax.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.cool)
ax.set_zlim(0, np.max(Z))
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.75, aspect=5)
plt.show()

# generate our population
Pop = 60
Domain = np.array( [[-4.0,4.0],[-4.0,4.0]] )
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
fig = plt.figure( figsize=(7, 7) )
ax = fig.gca()
cset = ax.contourf(X, Y, Z, cmap=cm.coolwarm, alpha=0.75)
ax.plot(Chromos[:,0], Chromos[:,1], 'ok')
plt.show()

# how many iterations do we run?
NoIterations = 50

# what percentage of our chromosomes do we mutate?
MutationRate = 0.4

# what is the crossover rate?
CrossoverRate = 0.5
CrossOverNumber = math.floor( Pop * CrossoverRate )

# storage for the new chromosomes at each generation
NewPopulation = np.zeros( (Pop,2) )
    
# generations
for i in range(NoIterations):
    
    # evaluate our chromsomes 
    Fitness = np.zeros(Pop)
    for j in range(Pop):
        Fitness[j] = A*2 + (Chromos[j,0]*Chromos[j,0] - A*np.cos(2*math.pi*Chromos[j,0])) + (Chromos[j,1]*Chromos[j,1] - A*np.cos(2*math.pi*Chromos[j,1]))
    
    # elitism
    best = np.argmin( Fitness )
    NewPopulation[0,0] = Chromos[best,0]
    NewPopulation[0,1] = Chromos[best,0]        
         
    # select which to cross and cross
    Selections = np.zeros((Pop,2))
    for j in range(1,CrossOverNumber+1):
        # pick the selectors 
        Selections[j,0] = int(RouletteWheelSelection(Fitness))
        Selections[j,1] = int(RouletteWheelSelection(Fitness))
        if( Selections[j,0] == Selections[j,1] ):
            Selections[j,1] = Selections[j,1] % Pop
        # crossover
        NewPopulation[j,0] = Chromos[ int(Selections[j,0]) , 0 ]
        NewPopulation[j,1] = Chromos[ int(Selections[j,1]) , 1 ]
        
    # rest are selected parents    
    for j in range(CrossOverNumber+1,Pop):
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
    Chromos = np.copy(NewPopulation)
    
    # plot
        
    ax.cla()
    cset = ax.contourf(X, Y, Z, cmap=cm.coolwarm, alpha=0.75)
    ax.plot(Chromos[:,0], Chromos[:,1], 'ok')
    ax.plot(Chromos[0,0], Chromos[0,1], 'xr', markersize=25)
    display(fig)
    clear_output(wait = True)
    plt.pause(0.2)
