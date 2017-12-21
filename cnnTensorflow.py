#Accuracy achieved = 97.17% with 5 epochs.
#import dependencies
from tqdm import tqdm
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#Get da data
mnist = input_data.read_data_sets("/data/mnist", one_hot=True)

#Defining hyperparameters
batch_size = 128 
leanring_rate = tf.constant(0.01,tf.float32)
epochs = 5
dropout_p = 0.75
classes = 10

#The dataset contains funneled images.
#So we have to reshape the dataset into a 28*28*1 image(3D tensor)
#Defining placeholders for input and output
X = tf.placeholder(tf.float32,[batch_size ,784],name='X')
Y = tf.placeholder(tf.float32,[batch_size ,10 ],name='Y')
dropout = tf.placeholder(tf.float32, name='dropout')

#Defining the model
#1st Convolutional layer( Convolve - ReLU - MaxPool )
#Different names for filters and biases in all the layers are for understading purpose, and are not necessarily required
with tf.variable_scope('CONV1'):
	images = tf.reshape(X, shape=[-1,28,28,1])
	filters = tf.get_variable('filter',[5,5,1,16], initializer=tf.truncated_normal_initializer())
	biases = tf.get_variable('biases',[16], initializer=tf.random_normal_initializer())
	conv = tf.nn.conv2d(images, filters,strides=[1,1,1,1],padding='SAME')#output dim = 28x28x16 activation maps(P=1)
	conv1 = tf.nn.relu(conv + biases, name='conv1')
	pool1 =tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')#output dim = 14x14x16

#2nd Convolutional layer
with tf.variable_scope('CONV2'):
	filters_2 = tf.get_variable('filters_2', shape=[5,5,16,32], initializer=tf.truncated_normal_initializer()) 
	biases_2= tf.get_variable('biases_2', shape=[32], initializer=tf.random_normal_initializer())
	conv_2 = tf.nn.conv2d(pool1, filters_2, strides=[1,1,1,1], padding='SAME')#output dim = 14x14x32(P=1)
	conv2 = tf.nn.relu(conv_2 + biases_2, name='conv2')
	pool2 = tf.nn.max_pool(conv2, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1],padding='SAME')#output dim = 14x14x32

#3rd Convolutional layer 
with tf.variable_scope('CONV3'):  
	filters_3 = tf.get_variable('filters_3', shape=[5,5,32,64], initializer=tf.truncated_normal_initializer()) 
	biases_3 = tf.get_variable('biases_3', shape=[64], initializer=tf.random_normal_initializer())
	conv_3 = tf.nn.conv2d(pool2, filters_3, strides=[1,1,1,1], padding='SAME')#output dim = 14x14x64(P=1)
	conv3 = tf.nn.relu(conv_3 + biases_3, name='conv3')
	pool3 = tf.nn.max_pool(conv3, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1],padding='SAME')#output dim = 14x14x64

#Fully connected layer
with tf.variable_scope('Flatten') as scope:
	final_fm_size = 14*14*64 #final feature maps flattening
	W = tf.get_variable('Weights_of_FC',[final_fm_size,1024],initializer=tf.truncated_normal_initializer())
	b = tf.get_variable('Biases_of_FC',[1024],initializer=tf.constant_initializer(0.0))
	new_pool3 = tf.reshape(pool3,shape=[-1,final_fm_size ])

	fully_connected = tf.nn.relu(tf.matmul(new_pool3,W) + b, name='Fully_connected_layer' )
	fully_connected = tf.nn.dropout(fully_connected, dropout_p, name='Dropout' )

with tf.variable_scope('softmax_linear') as scope:
	W = tf.get_variable('Weights', [1024, classes], initializer=tf.truncated_normal_initializer())
	b = tf.get_variable('biases', [classes], initializer=tf.random_normal_initializer())
	logits = tf.matmul(fully_connected, W) + b #get logits

#here cross entropy is the natural choice for loss calculation
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y)
loss = tf.reduce_mean(entropy, name='loss')
optimizer = tf.train.AdamOptimizer(leanring_rate).minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter("./visualize", sess.graph)
	num_batches = int(mnist.train.num_examples/batch_size)
	
	# I will update the code to add checkpoints, so for now bear with me :)
	print("\nGood To Go - Training has Started\n")
	for i in tqdm(range(epochs)):
		epoch_loss = 0
		for _ in range(num_batches):
			X_batch, Y_batch = mnist.train.next_batch(batch_size)
			e, _ = sess.run([loss, optimizer], feed_dict={X:X_batch,Y:Y_batch,dropout:dropout_p})
			epoch_loss += e
		print("The loss for EPOCH {0} is {1}".format(i+1,epoch_loss))
	
    #Testing model
	predictions = tf.nn.softmax(logits)
	correct_preds = tf.equal(tf.argmax(predictions,axis=1),tf.argmax(Y,axis=1))#axis=1 means that the operation is performed across the rows of predictions                                                                              
	accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

	num_batches = int(mnist.test.num_examples/batch_size)
	total = int(mnist.test.num_examples)
	overall_correct_preds = 0
	for _ in range(num_batches):
		X_batch, Y_batch = mnist.test.next_batch(batch_size)
		a = sess.run(accuracy, feed_dict={X:X_batch, Y:Y_batch}) 		
		overall_correct_preds += a
		print("Test Batch Accuracy ",a/batch_size)

	print("Accuracy of Model = {0}".format(100*(overall_correct_preds/total)))
	writer.close()

	print("Learning rate = {0}\nCross entropy loss fn(Natural choice)\nModel used Adaptive momentum optimizer to minimize loss\n".format(learning_rate))
	#Tensorboard --logdir = "visualize"
	Name = "visualize"
	print("To get the dataflow graph Use the command-> tensorboard --logdir=\"" + Name + "\" ")








 
    