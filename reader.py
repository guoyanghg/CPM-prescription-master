import tensorflow as tf

def read_data(file_queue):
    reader = tf.TextLineReader()
    key, value = reader.read(file_queue)
    defaults=[]
    #4210+6640
    for i in range(10850):
        defaults.append([0.])

    result = tf.decode_csv(value, defaults)
    return result

def create_pipeline(filename, batch_size, num_epochs=None):
    file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
    example = read_data(file_queue)

    min_after_dequeue = 20 #每次取数据后 保证队列中至少剩余的数据量
    capacity = min_after_dequeue+batch_size
    example_batch = tf.train.shuffle_batch(
        [example], batch_size=batch_size,capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )

    return example_batch

#Xbatch=create_pipeline('test.csv',50, 1)


#x=tf.placeholder('int32',[None,8694])
#init_op = tf.global_variables_initializer()
#local_init_op = tf.local_variables_initializer()  # local variables like epoch_num, batch_size
'''with tf.Session() as sess:
    sess.run(init_op)
    sess.run(local_init_op)

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Retrieve a single instance:
    try:
        #while not coord.should_stop():
        while True:
            example= sess.run(Xbatch)
            #print(example)
            data = sess.run(x,feed_dict={x:example})
            print(data)
    except tf.errors.OutOfRangeError:
        print ('Done reading')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()'''


