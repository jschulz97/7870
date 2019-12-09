from possi_flann_proj import *
import numpy as np

print('\nLoading Data...',end='')
mon_train_features = np.load('./data/monkey_train.npy')
mon_test_features  = np.load('./data/monkey_test.npy')
tig_train_features = np.load('./data/tiger_train.npy')
tig_test_features  = np.load('./data/tiger_test.npy')
che_train_features = np.load('./data/cheetah_train.npy')
che_test_features  = np.load('./data/cheetah_test.npy')
print('Finished.')

# bound_factors = [1,3,5]
# training_nums = [10,50,100,500]
bound_factors = np.arange(1, 10, .5)
training_nums = np.arange(10, 500, 20)
test_dim = 100
scores = []
scores1 = []

#Initialize a PKNN object
for tn in training_nums:
    for bf in bound_factors:
   
        PKNN   = Possi_FLANN(3,5,[tig_train_features,che_train_features],['tiger','cheetah'])
        score = PKNN.score([tig_test_features[:test_dim],che_test_features[:test_dim],mon_test_features[:test_dim]],['tiger','cheetah','none'],bf,tn)
        scores1.append(score)
    scores.append(scores1)
    scores1 = []

print('Z:',scores)