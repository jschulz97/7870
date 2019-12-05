import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import models, transforms

##
# Generate VAT Imagery given the provided features
# Features parameter should be 2D Matrix size (predictions , number of features)
class VAT_Image():
    def __init__(self,features,save_file='',desc=''):
        self.desc = desc
        self.save_file = save_file
        self.num_images = len(features)
        self.ivat = np.zeros((self.num_images,self.num_images))

        # Torch pairwise distance to generate differential matrix
        pdist = torch.nn.PairwiseDistance(p=2)
        for i in range(0,self.num_images):
            for j in range(0,self.num_images):
                self.ivat[i,j] = pdist(torch.tensor(features[i]).unsqueeze(0),torch.tensor(features[j]).unsqueeze(0))

        #Normalize
        mini = self.ivat.min()
        self.ivat -= mini
        maxi = self.ivat.max()
        self.ivat /= maxi

        #Create figure
        self.fig = plt.figure()
        ax = self.fig.add_subplot()
        ax.set_title(self.desc)
        heat = plt.imshow(self.ivat, cmap='hot')
        plt.colorbar(heat)

    ##
    # Do VAT Reordering
    def vat_reordering(self):
        old_ivat = self.ivat

        i,j = old_ivat.values()

        for r in range(2,self.num_images):
            j = np.min(old_ivat[r][:r-1])
            self.ivat[r][j] = old_ivat[r][j]


    ##
    # Normalize data and display heat map of differential matrix
    def save(self):
        if(self.save_file != ''):
            self.fig.savefig(self.save_file)
        else:
            print("\nNo save_file specified!\n")


    ##
    # Show plot
    def show(self):
        plt.show()

