import tensorflow as tf

x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

W = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = tf.multiply(X, W)

cost = tf.reduce_mean(tf.square(hypothesis - Y))

decent = W - tf.multiply(0.1, tf.reduce_mean(tf.multiply(tf.multiply(W, X) - Y, X)))
update = W.assign(decent)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print('w:', sess.run(W))

for step in range(100):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))
