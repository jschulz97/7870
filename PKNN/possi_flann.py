from pyflann import *
import numpy as np
from sklearn.neighbors import KDTree
from progressbar import progressbar

##
# Possibilistic KNN Classifier
# Needs to be fit with all known classes
class Possi_FLANN():
    def __init__(self,k=3,kfit=5):
        self.k      = k
        self.kfit   = kfit
        self.mean   = dict()
        self.sd     = dict()


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

