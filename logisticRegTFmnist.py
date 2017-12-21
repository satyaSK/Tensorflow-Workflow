#Import dependencies
import tensorflow as tf 
from tqdm import tqdm
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data

#Getting the data
mnist = input_data.read_data_sets('/data/mnist', one_hot=True)#Data

#defining hyperparameters
epochs = 50
learning_rate = tf.constant(0.001,tf.float32)
batch_size = 128

#Defining placeholders for input/output
with tf.name_scope("Data"):
	X = tf.placeholder(tf.float32,[batch_size,784], name='X')
	Y = tf.placeholder(tf.float32,[batch_size,10], name='Y')

#Defining trainable variables(Trainable = 'True')   
W = tf.Variable(tf.random_normal([784,10],stddev=0.1), name='Weights')#Rough intuition-contribution of each pixel to the each class
b = tf.Variable(tf.zeros([1,10]), name='Biases')


#Defining the model(We have one input layer and directly an output layer, so sorry no neural networks! :(
#But u can still make this a NN by adding one hidden layer(universal fn approximation)
#dont forget to initialize the weights and biases (in NN) otherwise all nodes will be dead(if ur using ReLU activation) 
logits = tf.matmul(X,W) + b 

#Defining loss fn reduced mean of (softmax -> cross_entropy) and optimizer(AdamOptimizer)
entropy = tf.nn.softmax_cross_entropy_with_logits( logits=logits, labels=Y, name='CE_loss')
loss = tf.reduce_mean(entropy)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


with tf.Session() as sess:
    # Train the Model
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter("./Visualize", sess.graph) 
	num_batches = int(mnist.train.num_examples/batch_size)
	print("\nGood To Go - Training Starts\n")
	for i in tqdm(range(epochs)):
	 	epoch_loss = 0
	 	for _ in range(num_batches):
	 		X_batch, Y_batch = mnist.train.next_batch(batch_size)
	 		e, _ = sess.run([loss, optimizer], feed_dict={X:X_batch,Y:Y_batch})
	 		epoch_loss += e
	 	print("The total loss for EPOCH {0} is {1}".format(i+1,epoch_loss))
	
	#Testing model
	predictions = tf.nn.softmax(logits)
	correct_preds = tf.equal(tf.argmax(predictions,axis=1),tf.argmax(Y,axis=1))#axis=1 means that the operation is performed across the rows of predictions                                                                              
	accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

	num_batches = int(mnist.test.num_examples/batch_size)
	overall_correct_preds = 0
	for _ in range(num_batches):
		X_batch, Y_batch = mnist.test.next_batch(batch_size)
		a = sess.run(accuracy, feed_dict={X:X_batch, Y:Y_batch}) 		
		overall_correct_preds += a
		print("Batch Accuracy ",a)

	print("Accuracy of Model = {0}".format(overall_correct_preds/mnist.test.num_examples))
	writer.close()

### Just for fun ###
	print("Learning rate = {0}\nCross entropy loss fn(Natural choice)\nModel used Adaptive momentum optimizer to minimize loss\n".format(learning_rate))
	#Tensorboard --logdir = "Visualize"
	Name = "Visualize"
	print("To get dataflow graph Use the command-> tensorboard --logdir=\"" + Name + "\" ")