# Convolutional Neural Networks
The above codes attempts to implement the CNN model using Tensorflow, and also Keras. Convolutional networks were first made popular by Yann Lecun, and they are basically used for tasks involving images as they capture feature representations more accurately as compared to other neural networks. I have discussed about the approach below

## Dependencies
* Tensorflow
* Numpy
* tqdm
* Keras

## Pipeline
```
Input Images -> Convolve(filters) -> Activation(ReLU) -> Max Pooling -> Flatten -> Fully Connected -> Logits -> Loss -> Backpropagate to update weights
```

## Dataflow Graph
![cnntensorflow](https://user-images.githubusercontent.com/34591573/34300433-0ad2c124-e74e-11e7-8c09-4af42f7cd5e6.png)

## Simplified Approach
* Data(funneled images in this case) is preprocessed into the required format.
* Randomly initiated filters(with size and stride) are made to convolve over the images(with appropriate padding) to capture feature representation and spacial information.
* The ReLU activation function is applied, followed by a Pooling(Max) layer.
* Once the filters have learnt to get excited about a feature(like an edge or simple slant line), they've learnt to get excited about a spacial position
* The output of the pooling layer is flattened and passed into the fully connected layer, which helps in finding non-linear patterns in the feature representations of the images.
* The output of the last hidden layer is passed through the softmax function which gives the probablity distribution over the classes.
* As always, with a prediction, comes a loss which we have to minimize using a loss function.
* Here the natural choice is the [Cross Entropy](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/) loss function.
* To minimize this loss, we use the Adaptive Momentum optimizer(aka [AdamOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer)) which, for our purpose, performs better than the classic gradient descent optimizer.
* The losses are calculated and back propagated(For us, Tensorflow is kind enough to care of that).
* Finally all the weights are updated in the direction of dereasing loss.
* During the training phase, we have to continuously take care that our model generalizes to all the data, and it does NOT overfit. For this we include Dropout.
* Smartly playing with hyperparameters, also helps the model to generalize and converge faster! 

## Basic Usage
For Running, type in terminal
```
cnnTensorflow.py
```
For the beautiful visualization, type in terminal
```
tensorboard --logdir="visualize"
```



