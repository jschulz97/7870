from mlp_mnist import MLP_MNIST

# def __init__(self,train_dim=5000, test_dim=5000, activation=tanh):
# def train(self,train_dim=0, eta=.0001, epoch=1, mini_batch_size=1):
# def test(self, test_dim=0, rand=True, ):

mlp = None

while(True):
    ## Training
    print('\n\nWelcome to the MLP_MNIST Companion App!')
    print('-- TRAINING MENU --')
    print('\nSelect an experiment to run from the following:')
    print('1. 2k Images, 20 Epochs, Mini-batches of 10 @ .01 Learning Rate')
    print('2. 2k Images, 5 Epochs, .01 Learning Rate')
    print('3. 2k Images, 20 Epochs, Mini-batches of 10 w/ Momentum of 0.25')
    print('0. Exit')
    print('>',end='')
    inp = int(float(input()))

    if(inp == 0):
        break

    elif(inp == 1):
        exp_desc='mini_batch'
        t = 2000
        w = .01
        p = 20
        b = 10
        e = .01
        m = 0
        mlp = MLP_MNIST(t,5000,exp_desc=exp_desc)
        d = 'tn='+str(t)+'_wsd='+str(w)+'_ep='+str(p)+'_mb='+str(b)+'_eta='+str(e)+'_m='+str(m)
        mlp.train(train_dim=t,eta=e,epoch=p,mini_batch_size=b, weight_init_sd=w, mom=m,desc=d)

    elif(inp == 2):
        exp_desc='no_batch'
        t = 2000
        w = .01
        p = 5
        b = 1
        e = .01
        m = 0
        mlp = MLP_MNIST(t,5000,exp_desc=exp_desc)
        d = 'tn='+str(t)+'_wsd='+str(w)+'_ep='+str(p)+'_mb='+str(b)+'_eta='+str(e)+'_m='+str(m)
        mlp.train(train_dim=t,eta=e,epoch=p,mini_batch_size=b, weight_init_sd=w, mom=m,desc=d)

    elif(inp == 3):
        exp_desc='momentum'
        t = 2000
        w = .01
        p = 20
        b = 10
        e = .01
        m = .25
        mlp = MLP_MNIST(t,5000,exp_desc=exp_desc)
        d = 'tn='+str(t)+'_wsd='+str(w)+'_ep='+str(p)+'_mb='+str(b)+'_eta='+str(e)+'_m='+str(m)
        mlp.train(train_dim=t,eta=e,epoch=p,mini_batch_size=b, weight_init_sd=w,mom=m, desc=d)

    ## Testing
    print('\n\n-- TESTING MENU --\n\nHow many images should I test on?')
    print('>',end='')
    inp = int(float(input()))
    mlp.test(inp)

    while(True):
        print('\nSelect from the list:')
        print('1. View training error plot')
        print('2. View confusion matrix')
        print('3. Show learned weights')
        print('0. Back to training menu')
        print('>',end='')
        inp = int(float(input()))

        if(inp == 0):
            break

        elif(inp == 1):
            mlp.plot_error()

        elif(inp == 2):
            mlp.show_cm()

        elif(inp == 3):
            print('\nSelect a weight (0-99):')
            print('>',end='')
            inp = int(float(input()))
            if(inp <= 99 and inp >= 0):
                mlp.show_weights(inp)

