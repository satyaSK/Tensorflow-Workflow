# Logistic Regression
The above code attempts to implement logistic regression using the Tensorflow framework.

## Dependencies
* Tensorflow
* Numpy
* tqdm

## Dataflow Graph
![logistic-regression](https://user-images.githubusercontent.com/34591573/34303575-048a3aaa-e75c-11e7-83b7-4675a0a88eae.png)
Notice how the data, weights and biases are passed into a model, which give out a prediction. Further the losses are calculated and the error is backpropagated.

## Simplified Approach
* As opposed to linear regression, logistic regression is used to predict continuous classses.
* The funneled inputs are fed into the model.
* The model here is simply the matrix multiplication of the inputs and the weights. Also we add a bias in the end.
* The outputs are the logits which we pass through the softmax function to get a probablity distribution over the classes(in this case 10 classes).
* Here the natural choice is the [Cross Entropy](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/) loss function and I found the [Adam optimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer) to do well in this setup.
* The predicted value (having highest probablity) is compared to the actual label and the error is backpropagated.
* The weights are then shifted in a direction such that the loss decreses.
* Notice that in this model, there is not a single hidden layer, so here we are NOT dealing with any neural network(So that gets rid of any confusion, if any).

## Basic Usage
For Running, type in terminal
```
python logisticRegTFmnist.py
```
I also included a dataflow graph, so for the beautiful visualization, type in terminal
```
tensorboard --logdir="Visualize"
```