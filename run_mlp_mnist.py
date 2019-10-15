from mlp_mnist import MLP_MNIST

mlp = None

# mlp = MLP_MNIST(2000,2000,)
# #train(self, train_dim=0, eta=.0001, epoch=1, ):
# mlp.train(eta=.00001, mini_batch_size=30)
# mlp.test(100)
#mlp.classify()

while(True):
    ## Training
    print('\n\nWelcome to the MLP_MNIST Companion App!')
    print('-- TRAINING MENU --')
    print('\nSelect an experiment to run from the following:')
    print('1. Default MLP')
    print('2. Mini-Batch')
    print('3. Momentum')
    print('0. Exit')
    print('>',end='')
    inp = int(float(input()))

    if(inp == 0):
        break

    elif(inp == 1):
        #def __init__(self,train_dim=5000, test_dim=5000, activation=tanh):
        mlp = MLP_MNIST(2000,2000,)

        #train(self, train_dim=0, eta=.0001, epoch=1, ):
        mlp.train(eta=.000001,epoch=10)

    elif(inp == 2):
        #def __init__(self,train_dim=5000, test_dim=5000, activation=tanh):
        mlp = MLP_MNIST(2000,2000,)

        #train(self, train_dim=0, eta=.0001, epoch=1, ):
        mlp.train(eta=.00001, mini_batch_size=30)

    elif(inp == 3):
        print('Momentum')

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

