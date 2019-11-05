# Set up

Install python packages:

    pip3 install python-mnist????

    pip3 install progressbar2

# Get MNIST Data

Once inside project folder, cd into the "data" directory:

    cd data/

Then download the data sets that this code is designed to use:

    wget https://jschulz.dev/mnist_train.csv

    wget https://jschulz.dev/mnist_test.csv

# Run

To run:

    python3 run_mlp_mnist.py
