# Artificial Neural Network
This code is implemented to demonstrate the architecture of the artificial neural network(aka ANN) using the tensorflow framework. I have not optimized the code for state-of-the-art results. But, this should be enough to get your basics on track.

## Dependencies
* Tensorflow
* Numpy
* tqdm

Use the ```pip install <package>``` command to install the dependencies.

## Pipeline
```
input batch -> hidden layers -> output -> loss -> backpropagate errors -> repeat
```

# Simplified Approach
* Firstly, We import our dependencies, setup the hyperparameters, and load the data(here, I'm using the MNIST dataset, which are funneled images, and the classes are one hot encoded).
* The input data is passed into the first hidden layer. So, for getting the values of the first hidden layer we simply multiply the weights(shape=[input_dim, hidden_1]) with the input, add a bias, and pass it through the activation function for introducing non-linearity(I have used ReLU for activation here). 
* Then we simply calculate the magnitude of deviation from the actual values(aka the truth), and then update the parameters(weights) in the right direction to minimize the loss.
* We keep on doing this iteratively until our model converges.
## Overfitting?
* Overfitting can be tackled in a number of ways, some of which are using the L1 and L2 regularization.
* Tuning the learning rate is the single most important knob which can get better results.
* Using Hinton's dropout to generalize the model also works!
* Play with the number of epochs. Evaluate the test performance at each epoch, and choose the best one(Early stopping).
* Experiment with updaters(aka optimizers), as different optimizers are well suited for different scenarios, for e.g try using RMSprop instead of plain SGD, or also momentum(Nestrovs).
* If your network is deep, you should probably be using Xaviers weight initializer(it basically prevents the vanishing gradient or exploding gradient problem from happening by initializing weights within an appropriate range, instead of random initialization) 

## Basic Usage
For Running, type in terminal
```
python annTensorflow.py
```




