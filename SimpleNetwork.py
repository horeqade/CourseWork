import tensorflow as tf
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os

class SimpleNetwork:
	def __init__(self, a, b):
		self.a = a
		self.b_data = b
		
		self.num_samples = 1000
		self.batch_size = 100
		
		self.X_data = np.random.uniform(1,10, (self.num_samples, 1))
		self.y_data = self.a * self.X_data + self.b_data + \
																np.random.normal(0,2, (self.num_samples, 1))
		
		self.X = tf.placeholder(tf.float32, shape = (self.batch_size, 1))
		self.y = tf.placeholder(tf.float32, shape = (self.batch_size, 1))
		
		with tf.variable_scope('linear_regression'):
			self.w = tf.Variable(tf.random_normal((1,1)))
			self.b = tf.Variable(tf.random_normal((1,)))
			
		self.y_pred = tf.matmul(self.X, self.w) + self.b
		self.loss = tf.reduce_mean((self.y - self.y_pred)**2)
		self.optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)
		self.sess = tf.Session()
		
		self.weights = [(1,0)]
		
	def train(self, num_steps):
		self.sess.run(tf.initialize_all_variables())
		for i in range(num_steps):
			indices = np.random.choice(self.num_samples, self.batch_size)
			X_batch, y_batch = self.X_data[indices], self.y_data[indices]
			_, loss_val, w_val, b_val = self.sess.run([self.optimizer, self.loss, self.w, self.b], 
															feed_dict = {self.X: X_batch, self.y: y_batch})
			if i % 50 == 0:
				print('epoch %d: loss %.5f, k=%.4f, b=%.4f' % (i//50, loss_val, w_val, b_val))
			if i % 10 == 0:
				self.weights.append((w_val, b_val))
	
	def predict(self, x):
		x_feed = (np.zeros((self.batch_size, 1)) + x).astype(np.float32)
		pred = self.sess.run(self.y_pred, feed_dict = {self.X: x_feed})
		print(pred[:len(x)])
	
	def update(self, i):
		label = 'epoch {0}, k = {1}, b = {2}'.format(i, self.weights[i][0], self.weights[i][1])
		true_label = 'f(x) = {0}x + {1} + noise'.format(self.a, self.b_data)
		
		line.set_ydata(self.weights[i][0]*self.X_data + self.weights[i][1])
		ax.set_xlabel(label)
		ax.set_ylabel(true_label)
		return line, ax
	
	def MakeGif(self, path):
		global fig, ax, line
		fig, ax = plt.subplots()
		fig.set_tight_layout(True)
		ax.scatter(self.X_data, self.y_data)
		line, = ax.plot(self.X_data, self.X_data * self.weights[0][0] + \
															self.weights[0][1], 'r-', linewidth=2)

		anim = FuncAnimation(fig, self.update, frames = np.arange(1, len(self.weights)), 
																interval = 100)
																
		if (os.path.isfile(path)):
			os.remove(path)
		print(path)
		anim.save(path, writer=PillowWriter(fps=24))
		
		