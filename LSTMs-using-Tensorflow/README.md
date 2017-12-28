# Long Short-term Memory(LSTM) Network
Everything in life is a sequence. Here, I've tried to create a simple LSTM model which functions on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). RNN's(or LSTM's) are really good at capturing sequences or the temporal dynamics of the data. It is a relatively simple implementation, and a really nice one for starting off. Along with many resources, Christoper Olah's [blogpost](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) helped me solidify my understanding even further :)

## Dependencies
* Tensorflow
* Numpy
* tqdm

# What's an LSTM bro?
* LSTM's are a variant of the vanilla recurrent neural networks, which help in tackling the vanishing/exploding gradient problem. There are many other methods used(like clipping) to reduce the effect of this problem, but LSTM's seem to outperform them all.
* LSTM's are used for learning long-term dependencies, which otherwise, are not possible with your usual recurrent nets. An LSTM unit achieves this by effectively using its input, output and forget gates(which are described in greater depth, in the blogpost I linked above).
* There is also a simple(but significantly changed) variant of the LSTM, called the GRU(gated recurrent unit), which is relatively less complex than an LSTM. The simple differece is that GRU's do not have the forget gate as opposed to LSTM's. These were created to reduce the complexity of the LSTM's(but in most cases LSTM's outperform GRU's almost all the time).

## Simplified Approach
* The images I'm dealing with are 28x28 funnelled images, which we want to firstly reshape into the required tensor, for our model to process.
* So now, we can perceive the images to be of 28 rows having 28 pixels per row.
* The 28 rows can be thought of as the timesteps by which we'll be unrolling the LSTM layer. So now after unrolling, each of the 28 timesteps(28 rows) will be taking in the 28 pixels belonging to each row. This will happen for batch_size number of images.
* Once we feed in the image, we only care about the output at the 28th timestep(that is, after the 28 pixels of the 28th row are hot and ready). At this point, the LSTM layer has stored a mixed composition of all the input it has seen so far.
* Now the output at the 28th timestep is taken, multiplied by a weight matrix(also add a bias), to give a prediction.
* These predictions are compared to the actulal values, the loss is calculated(cross-entropy), and then by using BPTT(backpropagation through time), the weights are updated in the right direction.
* I achieved a test accuracy of 98.76% with this simple LSTM model, I am still working on LSTM's and related hyperparameters, to try and better understand them, so do look for updates in the future.

## Basic Usage
For Running, type in terminal
```
python rnnTensorflow.py
```
For the beautiful visualization, type in terminal
```
tensorboard --logdir="Visualize"
```



