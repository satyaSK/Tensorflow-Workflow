import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn
from tqdm import tqdm

#get data
mnist = input_data.read_data_sets("/data/mnist", one_hot=True)

#define hyperparameters
batch_size = 128
time_steps = 28 #number of timesteps which take in input row
input_size = 28 #1 row has 28 pixels
num_units = 128 #number of intivisual LSTM units in LSTM cell
n_classes = 10
learning_rate = 0.01
epochs = 5
keep_prob = 0.5 #adding dropout

with tf.name_scope('Data'):
	X = tf.placeholder(tf.float32, [None, time_steps, input_size], name='X')
	Y = tf.placeholder(tf.float32, [None, n_classes], name='Y')

W2 = tf.Variable(tf.random_normal([num_units, n_classes]))
b2 = tf.Variable(tf.random_normal([n_classes]))

shapedInputs = tf.unstack(X, time_steps,1)# convert it into a list of tensor of shape [batch_size, input_size] of length time_steps which is the input taken by static_rnn

with tf.name_scope('LSTM'):
	cell = rnn.BasicLSTMCell(num_units, forget_bias = 1)# set the forgrt gate bias of the LSTM to 1(It is a default value but Im explicitly mentioning it)
	cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)#this is the LSTM layer wrapped with dropout
	all_outputs, all_hidden_states = rnn.static_rnn(cell, shapedInputs, dtype='float32')

prediction = tf.matmul(all_outputs[-1],W2) + b2

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels = Y)
loss = tf.reduce_mean(entropy)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

on_point=tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
training_accuracy=tf.reduce_mean(tf.cast(on_point,tf.float32))

with tf.Session() as sess:
    # Train the Model
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter("./Visualize", sess.graph) 
	num_batches = int(mnist.train.num_examples/batch_size)
	print("\nGood To Go - Training Starts\n")
	for i in tqdm(range(epochs)):
	 	epoch_loss = 0
	 	for j in range(num_batches):
	 		X_batch, Y_batch = mnist.train.next_batch(batch_size)
	 		X_batch = X_batch.reshape((batch_size,time_steps, input_size))#this is for getting the input in the size that the LSTM cell will accept
	 		e, _ = sess.run([loss, optimizer], feed_dict={X:X_batch,Y:Y_batch})
	 		epoch_loss += e
	 		if j%100==0:
	 			a = sess.run(training_accuracy, feed_dict={X:X_batch,Y:Y_batch})
	 			print("Training accuracy percent now: ",a*100)
	 	print("The total loss for EPOCH {0} is {1}".format(i+1,epoch_loss))
	
	#Testing the model

	num_batches = int(mnist.test.num_examples/batch_size)
	overall_correct_preds = 0
	for k in range(num_batches):
		X_batch, Y_batch = mnist.test.next_batch(batch_size)
		X_batch = X_batch.reshape((batch_size,time_steps, input_size))
		a = sess.run(training_accuracy, feed_dict={X:X_batch, Y:Y_batch}) 		
		overall_correct_preds += a
		print("Test {0} Batch Accuracy:{1}".format(k,a*100))

	print("Accuracy of Model = {0}".format((overall_correct_preds/num_batches)*100))
	writer.close()









