import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#read the data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#print the shape
print('Test shape:', mnist.test.images.shape)
print('Train shape:', mnist.train.images.shape)

#neural network parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

n_hidden_1 = 256 #first layer number of features
n_hidden_2 = 256 #second layer
n_input = 784 #data input, it is 28x28
n_classes = 10 #by 0 to 9

#Graph
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

#Model
def mpl(x, weights, biases):
	print('x:', x.get_shape(), 'W1:', weights['h1'].get_shape(), 'b1:', biases['b1'].get_shape())
	#Hidden layer with ReLU
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	layer_1 = tf.nn.relu(layer_1)
	#Hidden layer with ReLU
	print('layer_1:', layer_1.get_shape(), 'W2:', weights['h2'].get_shape(), 'b2:', biases['b2'].get_shape())
	layer_2 = tf.add(tf.matmul(layer_1,weights['h2']), biases['b2'])
	layer_2 = tf.nn.relu(layer_2)
	#Output layer with linear activation
	print('layer_2:', layer_2.get_shape(), 'W3:', weights['out'].get_shape(), 'b3:', biases['out'].get_shape())
	out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
	print('out_layer:', out_layer.get_shape())
	return out_layer

#store weights and biases
weights = {
	'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),	#784x256
	'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),  #256x256
	'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))   #256x10
}

biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden_1])),             #256x1
	'b2': tf.Variable(tf.random_normal([n_hidden_2])),             #256x1
	'out': tf.Variable(tf.random_normal([n_classes]))              #10x1
}

# Construct model
pred = mpl(x, weights, biases)

#Loss function and Optimizer (adam is an extension of gradient descent)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred,labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # To keep sizes compatible with model
    print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
