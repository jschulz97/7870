from pyflann import *
import numpy as np
from sklearn.neighbors import KDTree
from progressbar import progressbar
import time
import matplotlib.pyplot as plt

##
# Possibilistic KNN Classifier
# Needs to be fit with all known classes
class Possi_FLANN():
    def __init__(self,k,kfit,train_features,labels):
        self.k      = k
        self.kfit   = kfit
        self.mean   = dict()
        self.sd     = dict()
        self.train_features = train_features
        self.train_labels = labels

    def score(self,tf,tl,bf,tn,disp_cm=False):
        #Fit training data
        train_features = self.train_features[0][:int(tn)]
        train_labels   = np.array([self.train_labels[0]]*int(tn))
        first = True
        for c,l in zip(self.train_features,self.train_labels):
            if(first):
                first = False
            else:
                train_features = np.append(train_features,c[:int(tn)],axis=0)
                train_labels   = np.append(train_labels,np.array([l]*int(tn)),axis=0)
        
        test_features = tf[0]
        test_labels   = np.array([tl[0]]*tf[0].shape[0])
        first = True
        for c,l in zip(tf,tl):
            if(first):
                first = False
            else:
                test_features = np.append(test_features,c,axis=0)
                test_labels = np.append(test_labels,np.array([l]*c.shape[0]),axis=0)

        self.fit(train_features,train_labels)
        
        #Predict on test data
        full = []
        times = []
        #print('Predicting...')

        t1           = time.time()
        possi_scores = self.predict(test_features,bound_factor=bf)
        t2           = time.time()

        #full         = [v for k,v in possi_scores.items()]

        #print('Average time per prediction:',(t2-t1)/test_features.shape[0])

        #################################
        # build confusion matrix
        # (0,0) True class1
        # (0,1) False class2
        # (1,0) False class1
        # (1,1) True class2
        #################################
        classes = tl
        test_dim_all = tf[0].shape[0] * len(classes)
        classification = [''] * test_dim_all

        #Use scoring to assign classes
        for i in range(test_dim_all):
            score = 'none'
            max = 0
            for cls in self.classes:
                if(possi_scores[cls][i] == max):
                    score = 'none'
                elif(possi_scores[cls][i] > max):
                    score = cls
                    max = possi_scores[cls][i]

            classification[i] = score

        cm = np.zeros((len(classes),len(classes)))
        score = 0.0
        for i,fit in enumerate(classes):
            for j,test in enumerate(classes):
                for k,c in enumerate(classification):
                    if(fit == test_labels[k] and test == c):
                        cm[i,j] += 1
                        if(i == j):
                            score += 1.0

        if(disp_cm):
            mx = np.max(cm)

            fig = plt.figure()
            fig.suptitle('b: '+str(bf)+'  tn: '+str(tn))

            ax = fig.add_subplot()
            im = ax.imshow(cm,vmax=mx,vmin=0)

            for j in range(len(classes)):
                for k in range(len(classes)):
                    text = ax.text(k,j,cm[j,k],ha="center", va="center", color="w",fontsize=20)

            plt.colorbar(im)
            #plt.show()
            fig.savefig('./export/bf_'+str(bf)+'_tn_'+str(tn)+'.png')
        return score/float(len(classification))


    #################################
    # Fit PKNN to training data
    #################################
    def fit(self,X,Y):
        self.Y = Y
        self.train_set = X

        #Unique sorted list of classes
        self.classes = set(sorted(self.Y))

        self.flann = FLANN()
        #self.flann.build_index(self.train_set,algorithm="kmeans", branching=32,iterations=7,checks=16)
        self.flann.build_index(self.train_set)

        #For each class, find dists from each training case to every other training case
        for cls in self.classes:
            class_dists = []

            result, dists = self.flann.nn_index(self.train_set, self.kfit)

            #Only add dists from own class
            for i,d in enumerate(dists):
                if(self.Y[i] == cls):
                    class_dists.append(np.array(d).ravel())

            class_dists = [v[1:] for v in class_dists]
            class_dists     = np.ravel(np.array(class_dists))
            class_mean      = np.mean(class_dists)
            class_sd        = np.std(class_dists)
            self.mean[cls]  = class_mean
            self.sd[cls]    = class_sd
        #print(self.mean)
        #print(self.sd)


    #################################
    # Predict Possibilistic KNN
    # Predicts on set of test cases
    #################################
    def predict(self,test_set,bound_factor=1):
        results, dists = self.flann.nn_index(test_set, self.k)

        #For each case, classify using mean/sd
        #Get class for each

        possibilities = dict()
        for cls in self.classes:
            cls_possi = []
            for res,dis in zip(results,dists):
                poss = 0
                #for each case, we have 3 neighbors....
                #run through each neighbor and calculate if it's close enough to
                #the predicted class
                for r,d in zip(res,dis):
                    if(self.Y[r] == cls):
                        bound = bound_factor * self.sd[cls]
                        poss += (1/(1+np.power(np.maximum(0,d - bound),2)))

                cls_possi.append(round(poss,4))
            possibilities[cls] = cls_possi

        return possibilities

