#Import dependencies
import tensorflow as tf 
from tqdm import tqdm
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data

#Getting the data
mnist = input_data.read_data_sets('/data/mnist', one_hot=True)#Data

#defining hyperparameters
epochs = 40
learning_rate = tf.constant(0.001,tf.float32)
batch_size = 128
step = 10
validation_size = mnist.validation.images.shape[0]
print("Validation size = ",validation_size)
#Defining placeholders for input/output
with tf.name_scope("Data"):
	X = tf.placeholder(tf.float32,[None,784], name='X')
	Y = tf.placeholder(tf.float32,[None,10], name='Y')# usually put batch_size in place of 'None'

#Defining variables(Trainable = 'True')
def logistic_model(inputs):
	w_initialize = tf.truncated_normal_initializer()
	b_initialize = tf.constant_initializer(0)
	W = tf.get_variable('weigths',[784,10], initializer = w_initialize)#Rough intuition-contribution of each pixel to the each class
	b = tf.get_variable('Biases', [1,10], initializer = b_initialize)
	logits = tf.matmul(inputs, W) + b
	return logits
#Defining the model(We have one input layer and directly an output layer, so sorry no neural networks! :(
#But u can still make this a NN by adding one hidden layer(universal fn approximator)

#Defining loss fn reduced mean of (softmax -> cross_entropy) and optimizer(AdamOptimizer)
def calculate_loss(logits,targets):
	entropy = tf.nn.softmax_cross_entropy_with_logits( logits=logits, labels=targets, name='CE_loss')
	loss = tf.reduce_mean(entropy)
	return loss

def optimize(loss, leanring_rate):
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	return optimizer
def get_accuracy(predictions,targets):
	predictions = tf.nn.softmax(predictions)# first get probablity distribution
	correct_predictions = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(targets, axis=1))
	accuracy = tf.reduce_sum(tf.cast(correct_predictions,tf.float32))
	return accuracy 

#create model and return logits
logits = logistic_model(X)
#apply softmax and calculate the loss
loss = calculate_loss(logits,Y)
#optimize for an objective function
train_optimizer = optimize(loss, learning_rate)
#calculate accuracy
accuracy_operation = get_accuracy(logits,Y)
##### validation set(model has not seen this, meaning model is not trained on this set)
validation = {X:mnist.validation.images, Y:mnist.validation.labels}


with tf.Session() as sess:
    # Train the Model
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter("./Visualize", sess.graph) 
	num_batches = int(mnist.train.num_examples/batch_size)
	print("\nGood To Go - Training Starts\n")
	for i in rangetqdm((epochs+1)):
	 	epoch_loss = 0
	 	for _ in range(num_batches):
	 		X_batch, Y_batch = mnist.train.next_batch(batch_size)
	 		e, _,acc = sess.run([loss,train_optimizer,accuracy_operation], feed_dict={X:X_batch,Y:Y_batch})
	 		epoch_loss += e
	 	if i%step==0:
	 		val_acc = sess.run(accuracy_operation, feed_dict=validation)# Do validation
	 		print("Epoch {0} Training Accuracy: {1:.3f}".format(i,(acc/batch_size)*100))
	 		print("Validation accuracy now: {0:.3f}\n".format((val_acc/validation_size)*100))

	
	#Testing model
	num_batches = int(mnist.test.num_examples/batch_size)
	overall_correct_preds = 0
	for j in range(num_batches):
		X_batch, Y_batch = mnist.test.next_batch(batch_size)
		a = sess.run(accuracy_operation, feed_dict={X:X_batch, Y:Y_batch}) 		
		overall_correct_preds += a
		print("Batch {0} Accuracy:{1}".format(j,a/batch_size))

	print("Test accuracy of Model = {0}".format(overall_correct_preds/mnist.test.num_examples))
	writer.close()

### Just for fun ###
	print("Learning rate = {0}\nCross entropy loss fn(Natural choice)\nModel used Adaptive momentum optimizer to minimize loss\n".format(learning_rate.eval()))
	#Tensorboard --logdir = "Visualize"
	Name = "Visualize"
	print("To get dataflow graph Use the command-> tensorboard --logdir=\"" + Name + "\" ")