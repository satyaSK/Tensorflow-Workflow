## Write the readme
## figure out why softmax is not required
import tensorflow as tf 
import numpy as np 
from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data

# Get data
mnist = input_data.read_data_sets("/data/mnist", one_hot=True)


#Define Hyperparameters
learning_rate = 0.001
input_dim = 784
hidden_1= 1000
hidden_2= 1000
hidden_3= 500
classes = 10
batch_size = 128
epochs = 20


#Defining our placeholders
X = tf.placeholder(tf.float32,[None,input_dim],name='X')
Y = tf.placeholder(tf.float32,[None, classes],name='Y')


Weights = {'w1':tf.Variable(tf.truncated_normal([input_dim, hidden_1])),
           'w2':tf.Variable(tf.truncated_normal([hidden_1, hidden_2])),
           'w3':tf.Variable(tf.truncated_normal([hidden_2, hidden_3])),
           'w4':tf.Variable(tf.truncated_normal([hidden_3, classes]))}
 
Biases = {'b1': tf.Variable(tf.random_normal([hidden_1])),
          'b2': tf.Variable(tf.random_normal([hidden_2])),
          'b3': tf.Variable(tf.random_normal([hidden_3])),
          'b4': tf.Variable(tf.random_normal([classes]))}

#Define the model
def ANN(data):
    hidden_layer_1 = tf.nn.relu(tf.matmul(X, Weights['w1']) + Biases['b1'])
    hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_1, Weights['w2']) + Biases['b2'])
    hidden_layer_3 = tf.nn.relu(tf.matmul(hidden_layer_2, Weights['w3']) + Biases['b3'])
    output_layer_logits = tf.matmul(hidden_layer_3, Weights['w4']) + Biases['b4']
    return output_layer_logits

logits = ANN(X)
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
loss = tf.reduce_mean(entropy, name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    num_batches = int(mnist.train.num_examples/batch_size)

    for i in tqdm(range(epochs)):
        total_error = 0
        for _ in tqdm(range(num_batches)):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            _,e = sess.run([optimizer, loss], feed_dict={X:X_batch,Y:Y_batch})
            

            total_error += e
        print("Epoch {0} - Error {1}".format(i,total_error))

    #Testing
    predictions = tf.nn.softmax(ANN(X))    
    on_point = tf.equal(tf.argmax(predictions,axis=1),tf.argmax(Y,axis=1))
    accuracy = tf.reduce_sum(tf.cast(on_point, tf.float32))
    num_batches = int(mnist.test.num_examples/batch_size)
    n_examples = float(mnist.test.num_examples)
    overall_correct_preds = 0
    for _ in range(num_batches):
        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        a = sess.run(accuracy, feed_dict={X:X_batch, Y:Y_batch})         
        overall_correct_preds += a
        print("Test Batch Accuracy ",a/batch_size)
    print("total accuracy: {0}".format(overall_correct_preds/n_examples))