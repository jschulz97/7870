import ga_manual_pknn
import numpy as np

mutations = np.arange(0,1.1,.25)
crossovers = np.arange(0,1.1,.25)

mutations = np.array([.25,.25,.25,.5,.5,.5,.75,.75,.75])
crossovers = np.array([.25,.5,.75,.25,.5,.75,.25,.5,.75])

for mu,cr in zip(mutations,crossovers):
        ga_manual_pknn.do_ga(10,mu,cr,True)