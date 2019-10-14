from mlp_mnist import MLP_MNIST

mlp = None

while(True):
    ## Training
    print('\n\nWelcome to the MLP_MNIST Companion App!')
    print('Select an experiment to run from the following:\n')
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
        mlp.train(eta=.00001,)

    elif(inp == 2):
        #train(self, train_dim=0, eta=.0001, epoch=1, ):
        mlp.train(eta=.00001, mini_batch_size=30)

    elif(inp == 3):
        print('Momentum')

    ## Testing
    print('\nTesting...\nHow many images should I test on?')
    print('>',end='')
    inp = int(float(input()))
    mlp.test(inp)

    #while(True):

