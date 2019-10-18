from mlp_mnist import MLP_MNIST
import csv
from multiprocessing import *

max_proc_sem = Semaphore(4)
csv_sem      = Semaphore(1)

def do_mlp(mlp,t,e,p,b,w,d,m):
    mlp.train(train_dim=t,eta=e,epoch=p,mini_batch_size=b, weight_init_sd=w, desc=d, mom=m)
    mlp.test(1000)
    mlp.plot_error(show=False)
    score = mlp.show_cm(show=False)
    print('Network score:', score)
    mlp.plot_deltas(show=False)
    line = [score,d]
    csv_sem.acquire()
    #print("-----------ACQUIRED CSV--------------")
    with open(dire+exp_desc+'/results.csv','a') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(line)
    csv_sem.release()
    #print("-----------RELEASED CSV--------------")
    max_proc_sem.release()
    #print("-----------RELEASED PROC--------------")
    return

# def __init__(self,train_dim=5000, test_dim=5000, activation=tanh):
# def train(self,train_dim=0, eta=.0001, epoch=1, mini_batch_size=1):
# def test(self, test_dim=0, rand=True, ):

exps = [2]

dire = '/home/jschulz7/shared/'
#dire = './'

for i in exps:
    if(i == 1):
        exp_desc = 'shower_exp'
        mlp = MLP_MNIST(10000,1000,exp_desc=exp_desc,)

        tn      = [1000,5000,10000]
        weight  = [.1,.01,.001]
        epochs  = [1,10,50,100]
        batches = [1,50,500]
        eta     = [.01, .001, .0001, .00001]

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
                            print('Network score:', score)
                            mlp.plot_deltas(show=False)

                            line = [score,d]
                            with open(dire+exp_desc+'/results.csv','a') as csvfile:
                                spamwriter = csv.writer(csvfile, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                spamwriter.writerow(line)
    
    elif(i == 2):
        exp_desc = 'batch_exp'
        mlp = MLP_MNIST(10000,1000,exp_desc=exp_desc,)

        tn      = [5000]
        weight  = [.01]
        epochs  = [20]
        batches = [1,10]
        eta     = [.01,]
        mom     = [0]

        #p = [] * (len(tn) * len(weight) * len(epochs) * len(batches) * len(eta) * len(mom))
        proc = []

        for t in tn:
            for w in weight:
                for p in epochs:
                    for b in batches:
                        for e in eta:
                            for m in mom:
                                max_proc_sem.acquire()
                                #print("-----------ACQUIRED PROC--------------")
                                d = 'tn='+str(t)+'_wsd='+str(w)+'_ep='+str(p)+'_mb='+str(b)+'_eta='+str(e)+'_m='+str(m)
                                proc.append(Process(target=do_mlp, args=(mlp,t,e,p,b,w,d,m)))
                                proc[-1].start()
                                #do_mlp(mlp,t,e,p,b,w,d,m)
                                
                                
        for i in proc:
            i.join()
                                

    elif(i == 3):
        mlp = None
        mlp = MLP_MNIST(2000,2000,)
        #mlp.train(eta=.0001,epoch=1000,mini_batch_size=100)
        mlp.train(eta=.0001,epoch=30,mini_batch_size=1)
        mlp.test(1000)
        mlp.plot_error()
        mlp.show_cm()


