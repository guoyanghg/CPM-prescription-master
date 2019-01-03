from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import reader
import namedecoder
#%matplotlib inline


# Parameters
learning_rate = 0.1
training_epochs = 1
batch_size = 10


# Network Parameters
n_hidden_1 = 1024 # 1st layer num features
n_hidden_2 = 512 # 2nd layer num features
n_hidden_3 = 256 # 3rd layer num features
n_hidden_4 = 128 # 4th layer num features
n_input = 4210 #4210
n_output =6640

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    #'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    #'encoder_h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    #'decoder_h1': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_3*3])),
    #'decoder_h2': tf.Variable(tf.random_normal([n_hidden_3*3, n_hidden_2*3])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_output])),
    'one_step': tf.Variable(tf.random_normal([n_input, n_output])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    #'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    #'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),
    #'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3*3])),
    #'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2*3])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_output])),

}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))

    #layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   #biases['encoder_b3']))

    #layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['encoder_h4']),
                                   #biases['encoder_b4']))
    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))

    #layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                   #biases['decoder_b3']))

    #layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                   #biases['decoder_b4']))

    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

decoder_op= tf.add(tf.matmul(X, weights['one_step']),
                                  biases['decoder_b2'])

# Prediction
y_pred = decoder_op
softmax = tf.nn.softmax(y_pred)
# Targets (Labels) are the input data.
y_true = Y

# Define loss and optimizer, minimize the squared error
#cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
cost = -tf.reduce_mean(tf.multiply(y_true, tf.log(softmax)))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#get batch from reader
Xbatch = reader.create_pipeline('testdp.csv', 10, 1)

isTrain=True
# Initializing the variables
#init = tf.initialize_all_variables()
init_op = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()  # local variables like epoch_num, batch_size

# Launch the graph
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True



with tf.Session() as sess:
    #sess.run(init)
    sess.run(init_op)
    sess.run(local_init_op)
    saver=tf.train.Saver(max_to_keep=1)


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    # Training cycle
    try:
        # while not coord.should_stop():
        if isTrain:
            while True:
                example = sess.run(Xbatch)
                x=example[:, 0:4210]
                y=example[:, 4210:10850]
                loss, tp = sess.run([cost, optimizer], feed_dict={X: x, Y: y})
                print(loss)
        else:
            saver.restore(sess, 'ckpt/relu_softmax-ckpt')
            x=namedecoder.disease2onehot('G20')
            y=sess.run(y_pred, feed_dict={X: [x]})
            namedecoder.multihot2drugs(y)



    except tf.errors.OutOfRangeError:
        print('Done reading')
    finally:
        coord.request_stop()

    if isTrain:
        saver.save(sess,'ckpt/sm_softmax_reverse-ckpt')
    coord.join(threads)
    sess.close()


    print("Optimization Finished!")

