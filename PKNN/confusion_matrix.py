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

mon_train = './data/monkeys/train/'
mon_test  = './data/monkeys/test/'
tig_train = './data/tigers/train/'
tig_train = './data/tigers/test/'
che_train = './data/cheetahs/train/'
che_test  = './data/cheetahs/test/'


#Load features from file
mon_train_features = np.load('./data/monkeys/monkey_train.npy')
mon_test_features  = np.load('./data/monkeys/monkey_test.npy')
tig_train_features = np.load('./data/tigers/tiger_train.npy')
tig_test_features  = np.load('./data/tigers/tiger_test.npy')
che_train_features = np.load('./data/cheetahs/cheetah_train.npy')
che_test_features  = np.load('./data/cheetahs/cheetah_test.npy')

class1 = 'tiger'
class2 = 'cheetah'


#Get Features
#train_model = Get_Feature_Maps(data_dir=dir_train,model='resnet101',max_pooling=True)
#train_features = train_model.evaluate()
#test_model = Get_Feature_Maps(data_dir=dir_test,model='resnet101',max_pooling=True)
#test_features = test_model.evaluate()

bound_factor = [1,3,5]
training_num = [10,50,100,500]
#bound_factor = [3]
#training_num = [50]

test_dim = 100

# Change boundary factor (a * sd)
for bf in bound_factor:

    # Change number of images trained on
    for tn in training_num:
        #Initialize a PKNN object
        knn = 3
        KNN = Possi_FLANN(k=knn)

        #Get classes for training data
        #class_file = open(dir_train+"classes.dat","r")
        #y = class_file.read().splitlines()
        #ydim = len(y)
        #classes = sorted(set(y))

        #Fit training data
        train_features = np.append(tig_train_features[:tn],che_train_features[:tn],axis=0)
        test_features = np.append(tig_test_features[:test_dim],che_test_features[:test_dim],axis=0)



        y = np.append(np.array([class1 for i in range(tn)]),np.array([class2 for i in range(tn)]),axis=0)
        KNN.fit(train_features,y)
        y = np.append(np.array([class1 for i in range(test_dim)]),np.array([class2 for i in range(test_dim)]),axis=0)

        #Predict on test data
        full = []
        times = []
        print(' Predicting...')

        t1           = time.time()
        possi_scores = KNN.predict(test_features,bound_factor=bf)
        t2           = time.time()

        full         = [v for k,v in possi_scores.items()]

        #classes      = [k for k,v  in possi_scores.items()]
        #classes = set(classes)
        #print(classes)
        #print("tig mon",possi_scores['tiger'][:test_dim])
        #print("tig tig",possi_scores['tiger'][test_dim:])
        #print("mon tig",possi_scores['monkey'][:test_dim])
        #print("mon mon",possi_scores['monkey'][test_dim:])
        print('Average time per prediction:',(t2-t1)/64)

        #################################
        # build confusion matrix
        # (0,0) True class1
        # (0,1) False class2
        # (1,0) False class1
        # (1,1) True class2
        #################################
        cm = np.zeros((2,2))

        for i in range(200):
            if(possi_scores[class1][i] > possi_scores[class2][i]):
                score = class1
            elif(possi_scores[class1][i] < possi_scores[class2][i]):
                score = class2
            else:
                score = 'neither'

            if(score != 'neither'):
                if(score == y[i] and y[i] == class1):
                    cm[0,0] += 1
                elif(score != y[i] and y[i] == class1):
                    cm[0,1] += 1
                elif(score == y[i] and y[i] == class2):
                    cm[1,1] += 1
                else:
                    cm[1,0] += 1



        #for i,val in enumerate(possi_scores['tiger'][:test_dim]):
        #    if(not (val == 0 and possi_scores['monkey'][i] == 0)):
        #        #print(val,possi_scores['monkey'][i])
        #        if(val < possi_scores['monkey'][i]):
        #            cm[0,0]+=1
        #        else:
        #            cm[0,1]+=1

        #for i,val in enumerate(possi_scores['monkey'][test_dim:]):
        #    if(not (val == 0 and possi_scores['tiger'][i] == 0)):
        #        print(val,possi_scores['tiger'][i])
        #        if(val > possi_scores['tiger'][i]):
        #            cm[1,1]+=1
        #        else:
        #            cm[1,0]+=1

        mx = np.max(cm)

        fig = plt.figure()
        fig.suptitle('b = '+str(bf)+', tn = '+str(tn))

        ax = fig.add_subplot()
        im = ax.imshow(cm,vmax=mx,vmin=0)

        for j in range(2):
            for k in range(2):
                text = ax.text(k,j,cm[j,k],ha="center", va="center", color="w",fontsize=15)

        plt.colorbar(im)
        #plt.show()
        fig.savefig('cmexport/'+class1+'_'+class2+'_b_'+str(bf)+'_tn_'+str(tn)+'.png')


