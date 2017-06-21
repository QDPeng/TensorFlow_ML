import tensorflow as tf
import matplotlib.pyplot as plt

X = [1., 2., 3.]
Y = [1., 2., 3.]
m = m_samples = len(X)

W = tf.placeholder(tf.float32)
hypothesis = tf.multiply(X, W)

cost = tf.reduce_mean(tf.pow(hypothesis - Y, 2)) / m

W_val = []
cost_val = []

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print("hypothesis:", sess.run(hypothesis, feed_dict={W: -3}))
print("pow:", sess.run(tf.pow(hypothesis - Y, 2), feed_dict={W: -3}))
print("reduce_mean:", sess.run(tf.reduce_mean(tf.pow(hypothesis - Y, 2)), feed_dict={W: -3}))

for i in range(-30, 50):
    print(i * 0.1, sess.run(cost, feed_dict={W: i * 0.1}))
    W_val.append(i * 0.1)
    cost_val.append(sess.run(cost, feed_dict={W: i * 0.1}))

plt.plot(W_val, cost_val, "ro")
plt.ylabel("cost")
plt.xlabel("W")
plt.show()
