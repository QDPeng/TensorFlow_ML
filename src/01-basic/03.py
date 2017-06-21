import tensorflow as tf

a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)

add = tf.add(a, b)
mul = tf.multiply(a, b)

with tf.Session() as sess:
    print("Addition with variables: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
    print("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))

