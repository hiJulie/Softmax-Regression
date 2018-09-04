#准备数据
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

learning_rate = 0.5
training_epochs =1000
display_step = 50
save_step = 500

#forward
X = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x')
Y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y')

with tf.variable_scope('softmax_regression'):
	W = tf.get_variable(initializer=tf.zeros([784, 10]),name="W")
	b = tf.get_variable(initializer=tf.zeros([10]), name="b")
	y_hat = tf.nn.softmax(tf.matmul(X, W) + b)

#loss及loss优化
with tf.variable_scope('loss'):
	loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_hat), reduction_indices=[1]))
	train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#使用tensorboard
ckpt_path = './ckpt/softmax-regression-model.ckpt'
saver = tf.train.Saver()
summary_path = './ckpt/graph'
tf.summary.histogram('weights', W)
tf.summary.histogram('bias', b)
tf.summary.scalar('loss', loss)
merge_all = tf.summary.merge_all()

#训练
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	summary_writer = tf.summary.FileWriter(summary_path, sess.graph)
	x, y = mnist.train.next_batch(100)
	for epoch in range(training_epochs):
		_, summary = sess.run([train_op, merge_all], feed_dict={X:x, Y:y})
		if (epoch+1) % display_step == 0:
			c = sess.run(loss, feed_dict={X:x, Y:y})
			print('Epoch:', '%04d' % (epoch+1), 'loss=', '{:.9f}'.format(c), 'w=', W.eval(), 'b=', b.eval())
			summary_writer.add_summary(summary, global_step=epoch)
		if (epoch+1) % save_step == 0:
			save_path = saver.save(sess, ckpt_path, global_step=epoch)
			print('model saved in file: %s' % save_path)
	print('Optimization Finished!')
	save_path = saver.save(sess, ckpt_path, global_step=epoch)
	print('Final model saved in file: %s' % save_path)
	summary_writer.close()
	training_loss = sess.run(loss, feed_dict={X: x, Y: y})
	print('Training loss=', training_loss, 'w=', sess.run(W), 'b=', sess.run(b), '\n')
	
#测评准确率
	prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(y_hat, 1))
	accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
	print(sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))