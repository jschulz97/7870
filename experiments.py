from mlp_mnist import MLP_MNIST
import csv

# def __init__(self,train_dim=5000, test_dim=5000, activation=tanh):
# def train(self,train_dim=0, eta=.0001, epoch=1, mini_batch_size=1):
# def test(self, test_dim=0, rand=True, ):

exps = [1]

tn      = [1000,5000,10000]
weight  = [.1,.01,.001]
epochs  = [1,10,50,100]
batches = [1,50,500]
eta     = [.01, .001, .0001, .00001]

#dire = '/home/jschulz7/shared/shower_exp/'
dire = './'

for i in exps:
    if(i == 1):
        mlp = MLP_MNIST(10000,1000,exp_desc='shower_exp',)

        with open(dire+'results.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for t in tn:
                for w in weight:
                    for p in epochs:
                        for b in batches:
                            for e in eta:
                                d = 'tn='+str(t)+'_wsd='+str(w)+'_ep='+str(p)+'_mb='+str(b)+'_eta='+str(e)
                                mlp.train(train_dim=t,eta=e,epoch=p,mini_batch_size=b, weight_init_sd=w, desc=d)
                                mlp.test(1000)
                                mlp.plot_error(show=False)
                                score = mlp.show_cm(show=False)
                                mlp.plot_deltas(show=False)

                                line = [score,d]
                                spamwriter.writerow(line)

                            


        #mlp.train(eta=.0001,epoch=1000,mini_batch_size=100)
        mlp.train(eta=.0001,epoch=1,mini_batch_size=1, weight_init_sd=.01, desc='sd=.01')
        
    
    elif(i == 2):
        mlp = None
        mlp = MLP_MNIST(2000,2000,)
        #mlp.train(eta=.0001,epoch=1000,mini_batch_size=100)
        mlp.train(eta=.0001,epoch=30,mini_batch_size=1)
        mlp.test(1000)
        mlp.plot_error()
        mlp.show_cm()

    elif(i == 3):
        mlp = None
        mlp = MLP_MNIST(2000,2000,)
        #mlp.train(eta=.0001,epoch=1000,mini_batch_size=100)
        mlp.train(eta=.0001,epoch=30,mini_batch_size=1)
        mlp.test(1000)
        mlp.plot_error()
        mlp.show_cm()