# Optimizing PKNN with Genetic Algorithm

## Install

To install the required packages for this code:

    pip3 install matplotlib 
    pip3 install numpy
    pip3 install progressbar2

## How to Use

Test file for quick start is in test_ga.py.

### Public methods in ga_manual_pknn.py:

#### do_ga(*iterations*, *mutation*, *crossover*, *disp*)

*iterations:* int, Number of iterations for genetic algorithm

*mutation:* double, Mutation rate

*crossover:* double, Crossover rate

*disp:* bool, display graphs/confusion matrices

Executes Genetic Algorithm to optimize PKNN parameters. 

### Public methods in possi_flann_proj.py:

#### init(*k*, *kfit*, *train_features*, *labels*)

*k:* int, # Nearest Neighbors to compute on in execution

*kfit:* int, # Nearest Neighbors to compute on in fitting

*train_features:* List, list of numpy matrices (feature activations of CNN)

*labels:* List, list of Strings, labels corresponding to each passed training feature matrix

#### score(*tf*, *tl*, *bf*, *tn*, *disp_cm*)

*tf:* List, list of numpy matrices (feature activations of CNN)

*tl:* List, list of labels corresponding to each passed testing feature matrix

*bf:* double, Boundary factor 

*tn:* int, # of training images to use when fitting the PKNN

*disp_cm:* bool, display/save the confusion matrix output of the GA




