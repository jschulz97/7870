########################################
# Changes num of training images
# for different bound factors
########################################

from possi_flann import *
from get_feature_maps import *
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import progressbar
import time
from generate_vat import *

mon_fit = './data/monkeys/train/'
mon_test  = './data/monkeys/test/'
tig_fit = './data/tigers/train/'
tig_train = './data/tigers/test/'
che_fit = './data/cheetahs/train/'
che_test  = './data/cheetahs/test/'

#Load features from file
fit_features = {}
test_features = {}
fit_features['monkey']    = np.load('./data/monkeys/monkey_train.npy')
test_features['monkey']     = np.load('./data/monkeys/monkey_test.npy')
fit_features['tiger']     = np.load('./data/tigers/tiger_train.npy')
test_features['tiger']      = np.load('./data/tigers/tiger_test.npy')
fit_features['cheetah']   = np.load('./data/cheetahs/cheetah_train.npy')
test_features['cheetah']    = np.load('./data/cheetahs/cheetah_test.npy')

classes = ['tiger','cheetah','monkey']

bound_factor = [1,3,5]
fitting_num = [10,50,100,500]
#bound_factor = [3]
#training_num = [50]

test_dim = 100

# Change boundary factor (a * sd)
for bf in bound_factor:

    # Change number of images trained on
    for fit_dim in fitting_num:
        #Initialize a PKNN object
        knn = 3
        KNN = Possi_FLANN(k=knn)

        #Create fit/test data
        fit_feat_all = fit_features[classes[0]][:fit_dim]
        for cls in classes:
            if(cls != classes[0]):
                fit_feat_all = np.append(fit_feat_all,fit_features[cls][:fit_dim],axis=0)

        test_feat_all = test_features[classes[0]][:test_dim]
        for cls in classes:
            if(cls != classes[0]):
                test_feat_all = np.append(test_feat_all,test_features[cls][:test_dim],axis=0)

        #Create fitting labels
        d = []
        for cls in classes:
            d = np.append(d,[cls for i in range(fit_dim)],axis=0)

        #Fit PKNN
        KNN.fit(fit_feat_all,d)

        #Create validation labels
        y = []
        for cls in classes:
            y = np.append(y,[cls for i in range(test_dim)],axis=0)


        #Predict on test data
        full = []
        times = []
        print(' Predicting...')

        t1           = time.time()
        possi_scores = KNN.predict(test_feat_all,bound_factor=bf)
        t2           = time.time()

        full         = [v for k,v in possi_scores.items()]

        print('Average time per prediction:',(t2-t1)/64)

        test_dim_all = test_dim * len(classes)
        classification = [''] * test_dim_all

        #Use scoring to assign classes
        for i in range(test_dim_all):
            score = 'neither'
            max = 0
            for cls in classes:
                if(possi_scores[cls][i] == max):
                    score = 'neither'
                elif(possi_scores[cls][i] > max):
                    score = cls
                    max = possi_scores[cls][i]

            classification[i] = score


        #################################
        # build confusion matrix
        # (0,0) True class1
        # (0,1) False class2
        # (1,0) False class1
        # (1,1) True class2
        # etc...
        #################################
        cm = np.zeros((len(classes),len(classes)))

        for i,fit in enumerate(classes):
            for j,test in enumerate(classes):
                for k,c in enumerate(classification):
                    if(fit == y[k] and test == c):
                        cm[i,j] += 1

        mx = np.max(cm)

        fig = plt.figure()
        fig.suptitle('b = '+str(bf)+', tn = '+str(fit_dim))

        ax = fig.add_subplot()
        im = ax.imshow(cm,vmax=mx,vmin=0)

        for j in range(len(classes)):
            for k in range(len(classes)):
                text = ax.text(k,j,cm[j,k],ha="center", va="center", color="w",fontsize=15)

        plt.colorbar(im)
        #plt.show()

        class_string = ''
        for cls in classes:
            class_string += (cls+'_')

        fig.savefig('cmexport/'+class_string+'b_'+str(bf)+'_fn_'+str(fit_dim)+'.png')


