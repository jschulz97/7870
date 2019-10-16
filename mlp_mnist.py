# libs
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from IPython.display import clear_output
import sys
import random
from progressbar import progressbar
import os

#Init random seed
random.seed(0)

#################
# sigmoid
def sigmoid(x, derive=False): 
    if derive: 
        return x * (1.0 - x) 
    return ( 1.0 / (1.0 + np.exp(-x)) )

#################
# tanh
def tanh(x, derive=False):
    if derive:
        return np.power(1/np.cosh(x),2)
    return np.tanh(x)


##########################################################
# MLP_MNIST class will hold all mlp nn data and methods
class MLP_MNIST:
    def __init__(self,train_dim=5000, test_dim=5000, activation=tanh, exp_desc=''):
        ## data set
        self.actfx        = activation
        self.train_dim    = train_dim
        self.test_dim     = test_dim
        self.train_data   = np.loadtxt('./data/mnist_train.csv',delimiter=',',max_rows=self.train_dim)
        self.test_data    = np.loadtxt('./data/mnist_test.csv',delimiter=',',max_rows=self.test_dim)

        self.train_labels = self.train_data[:,0]
        self.train_data   = self.train_data[:,1:]
        self.train_data   = np.reshape(self.train_data,(self.train_dim,28,28))
        self.train_data   = self.train_data[:,::2,::2] / 255.0
        #train_data   = train_data[:,::2,::2] * (.99 / 255.0) + .01 ??
        self.train_data   = np.reshape(self.train_data,(self.train_dim,196))

        self.test_labels  = self.test_data[:,0]
        self.test_data    = self.test_data[:,1:]
        self.test_data    = np.reshape(self.test_data,(self.test_dim,28,28))
        self.test_data    = self.test_data[:,::2,::2] / 255.0
        self.test_data    = np.reshape(self.test_data,(self.test_dim,196))

        #Fix labels to one-hot encoding
        tl = np.zeros((self.train_dim,10))
        for i,l in enumerate(self.train_labels):
            tl[i][int(l)] = 1
        self.train_labels = tl

        tl = np.zeros((self.test_dim,10))
        for i,l in enumerate(self.test_labels):
            tl[i][int(l)] = 1
        self.test_labels = tl

        self.did_i_train = False
        self.did_i_test  = False

        #create dir for saving pics
        #self.cwd = '/home/jschulz7/shared/'+exp_desc
        self.cwd = './'+exp_desc
        try:
            os.mkdir(self.cwd)
        except:
            print('Directory',self.cwd,'exists.')


    #####################################
    # Training! Can alter training dims 
    def train(self,train_dim=0, eta=.0001, epoch=1, mini_batch_size=1, weight_init_sd=.1, desc='', ):
        self.eta   = eta    # learning rate
        self.epoch = epoch
        self.desc  = desc
        v1         = np.ones((1,101))
        o          = np.ones((1,10))

        # weights with random numbers
        self.h1_weights  = np.random.normal(0, weight_init_sd, size=(100,197))
        self.out_weights = np.random.normal(0, weight_init_sd, size=(10,101))
        self.h1_delta_full = np.array([])
        self.ow_delta_full = np.array([])

        #Find train_dim
        if(train_dim == 0):
            train_dim = self.train_dim
        elif(train_dim > self.train_dim):
            print("\n\nWarning! Number of training images ("+str(train_dim)+") too large. Increase train_dim parameter (currently "+str(self.train_dim)+") on object initialization.")
            train_dim = self.train_dim
        print("\nTraining on",train_dim,"images.")

        #self.err = np.zeros((self.epoch,int(train_dim/mini_batch_size)))  # init error 
        self.err = []

        ## Epochs
        for k in range(self.epoch): 
            #rand index list
            ind = []
            print('\nTraining Epoch #'+str(k)+'...')

            #Handle mini-batching or not
            #If mini-batching, outside for loop should only run once
            li = 0
            if(mini_batch_size != 1):
                li = 1
            else:
                li = train_dim

            ## Input Data
            for l in progressbar(range(li)):
                delta_ow = np.zeros((mini_batch_size,10))
                delta_h1 = np.zeros((mini_batch_size,100))

                ## Mini-Batch (defaults to 1 = no mini-batching)
                for b in range(mini_batch_size):
                    # Get random index
                    i = np.random.randint(low=0, high=self.train_dim, )
                    while(i in ind):
                        i = np.random.randint(low=0, high=self.train_dim, )
                    ind.append(i)

                    x = self.train_data[i]

                    ## Forward pass
                    #   (1,100)          (1,197)          (197,1)    
                    for j in range(100): 
                        v1[0][j] = np.dot(np.append(x,1), np.transpose(self.h1_weights[j]))
                        v1[0][j] = self.actfx(v1[0][j])

                    #   (1,10)          (1,101)  (101,1)    
                    for j in range(10):
                        o[0][j] = np.dot(v1,     np.transpose(self.out_weights[j]))
                        o[0][j] = o[0][j]
                    
                    ## Error
                    #self.err[k][l] = np.sum(((1.0/2.0) * np.power((o.T - self.train_labels[i]), 2.0)))
                    self.err.append(np.sum((1.0/2.0) * np.power((o.T - self.train_labels[i]), 2.0)))

                    ## Backprop
                    # Output layer
                    #(1,10)                  (10,1)                                       (10,1) 
                    delta_ow[b]    = np.reshape((-1.0) * (np.array(self.train_labels[i]) - np.reshape(o,((10,)))), (10,1) ).T
                    
                    # Layer 1
                    #(1,100)        (1,10)             (10,100)                         (1,100)
                    delta_h1[b]    = np.dot(delta_ow[b], self.out_weights[:,:100]) * self.actfx(v1[:,:100],derive=True)

                ## Aggregate batch results
                delta_ow_batch = np.array([np.sum(delta_ow[:,col]) for col in range(10)]) / mini_batch_size
                delta_h1_batch = np.array([np.sum(delta_h1[:,col]) for col in range(100)]) / mini_batch_size

                self.h1_delta_full = np.append(self.h1_delta_full, delta_h1_batch)
                self.ow_delta_full = np.append(self.ow_delta_full, delta_ow_batch)

                #shrink eta through the epochs
                new_eta = self.eta / np.power(10, int(k/10))

                ## update rule
                # Output layer 
                for j in range(10):
                    self.out_weights[j] -= new_eta * v1.ravel() * delta_ow_batch[j]
                
                # Hidden layer 1
                for j in range(100):
                    self.h1_weights[j] -= new_eta * np.append(x,1) * delta_h1_batch[j]

        self.did_i_train = True


    ###############################################
    # Testing
    def test(self, test_dim=0, rand=True, ):
        #Find test_dim
        if(test_dim == 0):
            test_dim = self.test_dim
        elif(test_dim > self.test_dim):
            print("\n\nWarning! Number of testing images ("+str(test_dim)+") too large. Increase test_dim parameter (currently "+str(self.test_dim)+") on object initialization.")
            test_dim = self.test_dim
        print("\nTesting on",test_dim,"images...") 
        self.test_dim = test_dim

        self.pred_res  = np.zeros((test_dim,10))   
        self.pred_ind  = np.zeros((test_dim,))

        v1  = np.ones((1,101))
        o   = np.ones((1,10))
        ind = []

        ## Input Data
        for l in progressbar(range(test_dim)):
            # Can decide with 'rand' parameter to test in order
            if(rand):
                # Get random index
                i = np.random.randint(low=0, high=self.test_dim, )
                while(i in ind):
                    i = np.random.randint(low=0, high=self.test_dim, )
                ind.append(i)
            else:
                i = l

            x = self.test_data[i]

            ## Forward pass
            #   (1,100)          (1,197)          (197,1)    
            for j in range(100): 
                v1[0][j] = np.dot(np.append(x,1), np.transpose(self.h1_weights[j]))
                v1[0][j] = self.actfx(v1[0][j])

            #   (1,10)          (1,101)  (101,1)    
            for j in range(10):
                o[0][j] = np.dot(v1,     np.transpose(self.out_weights[j]))
                o[0][j] = o[0][j]
            
            self.pred_res[l] = o[0]
            self.pred_ind[l] = i

        self.did_i_test = True
        self.classify()


    #############################
    # Classifier
    def classify(self, ):
        #print(self.pred_res[:10])

        for i in range(self.test_dim):
            mxi = np.argmax(self.pred_res[i])
            self.pred_res[i]      = np.zeros((10,))
            self.pred_res[i][mxi] = 1

        # for i,x in enumerate(self.pred_res[:10]):
        #     ind = int(self.pred_ind[i])
        #     print(self.test_labels[ind])
        # print(self.pred_res[:10])


    #############################
    # Plot error over updates
    def plot_error(self, show=True):  
        if(self.did_i_train):
            fig = plt.figure(figsize=(11,9))
            plt.plot(self.err)
            plt.ylabel('error')
            plt.xlabel('updates')
            print("\nDisplaying error plot...\n")
            plt.savefig(self.cwd+'/error_plot_'+self.desc+'.jpg')
            if(show):    
                plt.show()
        else:
            print('\nTrain the network first!\n')


    #############################
    # Plot deltas over updates
    def plot_deltas(self, show=True):  
        if(self.did_i_train):
            fig = plt.figure(figsize=(11,9))
            plt.plot(self.h1_delta_full.ravel())
            plt.ylabel('deltas')
            plt.xlabel('updates')
            print("\nDisplaying delta plot...\n")
            plt.savefig(self.cwd+'/deltah1_plot_'+self.desc+'.jpg',bbox_inches='tight')
            if(show):    
                plt.show()
            plt.cla()
            plt.plot(self.ow_delta_full.ravel())
            plt.savefig(self.cwd+'/deltaow_plot_'+self.desc+'.jpg',bbox_inches='tight')
            if(show):
                plt.show()
        else:
            print('\nTrain the network first!\n')

    
    #############################
    # Show confusion matrix
    def show_cm(self, show=True):
        if(self.did_i_test):
            cm = np.zeros((10,10))

            good = 0
            for i in range(self.test_dim):
                ind = int(self.pred_ind[i])
                tpi = np.argmax(self.test_labels[ind])
                cm[tpi] += self.pred_res[i]

                #score
                if(tpi == np.argmax(self.pred_res[i])):
                    good += 1

            mx = np.max(cm)

            fig = plt.figure(figsize=(9,9))
            fig.suptitle('')

            ax = fig.add_subplot()
            im = ax.imshow(cm,vmax=mx,vmin=0)

            for j in range(10):
                for k in range(10):
                    text = ax.text(k,j,cm[j,k],ha="center", va="center", color="w",fontsize=10)

            plt.colorbar(im)
            plt.ylabel('Actual')
            plt.xlabel('Prediction')
            plt.xticks(np.arange(0, 10, 1))
            plt.yticks(np.arange(0, 10, 1))
            ax.set_ylim(10-0.5, -0.5)
            print("\nDisplaying confusion matrix...\n")
            plt.savefig(self.cwd+'/cm_mat_'+self.desc+'.jpg')
            if(show):    
                plt.show()
            
            return (good/self.test_dim)
        else:
            print('\nTest on the network first!')
            return 0

    
    #############################
    # Show an image of one neuron's learned weights
    def show_weights(self, i=0, show=True):
        if(self.did_i_train):
            fig = plt.figure(figsize=(9,9))
            weights = self.h1_weights[i][:-1]
            weights = np.reshape(weights,(14,14))
            print("\nDisplaying learned weights...\n")
            plt.imshow(weights)
            plt.savefig(self.cwd+'/weights_img_'+self.desc+'.jpg')
            if(show):
                plt.show()
        else:
            print('\nTrain the network first!\n')