# Logistic Regression v2
The above code attempts to implement logistic regression using the Tensorflow framework. There are 2 versions of the logistic regression code which I've put up. One is the basic version, and the second vaersion has the following changes:
* I have encapsulated all the operations within functions, so they can be reused(it also improves readability alot).
* There is a validation set included in the second version, which helps in giving an idea about overfitting(so we can choose optimal hyperparameters).

## Dependencies
* Tensorflow
* Numpy

## Dataflow Graph
![logistic-regression](https://user-images.githubusercontent.com/34591573/34303575-048a3aaa-e75c-11e7-83b7-4675a0a88eae.png)
Notice how the data, weights and biases are passed into a model, which give out a prediction. Further the losses are calculated and the error is backpropagated.

# Simplified Approach
* As opposed to linear regression, logistic regression is used to predict continuous classses.
* The funneled input images are fed into the model.
* The model here is simply the matrix multiplication of the inputs and the weights. Also we add a bias in the end.
* The outputs we get are the logits, which we pass through the softmax function to get a probablity distribution over the classes(in this case 10 classes).
* Here the natural choice is the [Cross Entropy](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/) loss function and I found the [Adam optimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer) to do well in this setup.
* The predicted value (having highest probablity) is compared to the actual label, the loss is calculated and the error is backpropagated.
* The weights are then updated in the right direction, so as to minimize our objective function(loss function).
* Notice that in this model, there is not a single hidden layer, so here we are NOT dealing with any neural network(So that gets rid of any confusion, if any).
## Validation set?
* At regular intervals during training(and before testing) we feed into the model, what is called the "validation set".
* This validation set will give us an idea about how well the model is performing on the data which it hasn't seen. 
* As opposed to the testing set, the validation set helps us get an idea about when the model has started to overfit the data, giving an indication to abort the training. This can be seen at the point where the training accuracy seems to be increasing, but the validation accuracy has frozen or might even start to decrease!
* We want to stop our training at this point because we want our model to generalize to the data, to find meaningful patterns and not overfit and simply memorize the data.
![validation](https://user-images.githubusercontent.com/34591573/34531542-a57540ba-f0d8-11e7-8361-445aeb7b857d.png)

* Here, the validation accuracy I got after ```epoch = 30``` has remained almost unchanged, while the training accuracy has increased by almost 5% at ```epoch = 40```
* This shows that we should stop our training at ```epoch = 30``` itself, because from here onwards, our model will have started to overfit the data.
* Now thanks to the validation set, we can go back to the model and reset our hyperparameter ```epochs = 30```, and retrain our model to find generalized patterns in our data.

## Basic Usage
For Running, type in terminal
```
python logisticReg_v2.py
```
I also included a dataflow graph, so for the beautiful visualization, type in terminal
```
tensorboard --logdir="Visualize"
```