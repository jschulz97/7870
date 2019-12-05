from get_feature_maps import *

models = ['resnet18','resnet101','alexnet','google','vgg']
maxpool = ['False','False','False','True','True','True']

#models = ['resnet18','resnet50','resnet101']
#maxpool = ['False','False','False']

for md,mx in zip(models,maxpool):
    #Get_Feature_Maps(data_dir='./data8/',model=md,max_pooling=mx)
    Get_Feature_Maps(data_dir='./data/cheetahs/test/',model=md,max_pooling=True,save_features=True,save_file='./data/7870features/'+md+'_cheetahs_test').evaluate()
    Get_Feature_Maps(data_dir='./data/tigers/test/',model=md,max_pooling=True,save_features=True,save_file='./data/7870features/'+md+'_tigers_test').evaluate()
    Get_Feature_Maps(data_dir='./data/monkeys/test/',model=md,max_pooling=True,save_features=True,save_file='./data/7870features/'+md+'_monkeys_test').evaluate()
