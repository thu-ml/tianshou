import tensorflow as tf
import numpy as np
import time
import multi_gpu
import tensorflow.contrib.layers as layers

data = np.load("data.npz")
boards = data["boards"]
wins = data["wins"]
ps = data["ps"]
print (boards.shape)
print (wins.shape)
print (ps.shape)
def residual_block(input, is_training):
	normalizer_params = {'is_training': is_training,
						 'updates_collections': None}
	h = layers.conv2d(input, 256, kernel_size=3, stride=1, activation_fn=tf.nn.relu,
					  normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params,
					  weights_regularizer=layers.l2_regularizer(1e-4))
	residual = layers.conv2d(h, 256, kernel_size=3, stride=1, activation_fn=tf.identity,
							 normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params,
							 weights_regularizer=layers.l2_regularizer(1e-4))
	h = h + residual
	return tf.nn.relu(h)

def policy_heads(input, is_training):
	normalizer_params = {'is_training': is_training,
						 'updates_collections': None}
	h = layers.conv2d(input, 2, kernel_size=1, stride=1, activation_fn=tf.nn.relu,
					  normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params,
					  weights_regularizer=layers.l2_regularizer(1e-4))
	h = layers.flatten(h)
	h = layers.fully_connected(h, 362, activation_fn=tf.identity, weights_regularizer=layers.l2_regularizer(1e-4))
	return h

def value_heads(input, is_training):
	normalizer_params = {'is_training': is_training,
						 'updates_collections': None}
	h = layers.conv2d(input, 2, kernel_size=1, stride=1, activation_fn=tf.nn.relu,
					  normalizer_fn=layers.batch_norm, normalizer_params=normalizer_params,
					  weights_regularizer=layers.l2_regularizer(1e-4))
	h = layers.flatten(h)
	h = layers.fully_connected(h, 256, activation_fn=tf.nn.relu, weights_regularizer=layers.l2_regularizer(1e-4))
	h = layers.fully_connected(h, 1, activation_fn=tf.nn.tanh, weights_regularizer=layers.l2_regularizer(1e-4))
	return h


x = tf.placeholder(tf.float32,shape=[None,19,19,17])
is_training = tf.placeholder(tf.bool, shape=[])
z = tf.placeholder(tf.float32, shape=[None, 1])
pi = tf.placeholder(tf.float32, shape=[None, 362])

h = residual_block(x, is_training)
for i in range(18):
	h = residual_block(h, is_training)
v = value_heads(h, is_training)
p = policy_heads(h, is_training)
loss = tf.reduce_mean(tf.square(z-v)) - tf.reduce_mean(tf.multiply(pi, tf.log(tf.nn.softmax(p, 1))))
reg = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
total_loss = loss + reg
train_op = tf.train.RMSPropOptimizer(1e-2).minimize(total_loss)

var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
saver = tf.train.Saver(max_to_keep=10, var_list=var_list)
epochs = 100
batch_size = 32
batch_num = boards.shape[0] // batch_size
result_path = "./results/"
with multi_gpu.create_session() as sess:
	sess.run(tf.global_variables_initializer())
	ckpt_file = tf.train.latest_checkpoint(result_path)
	if ckpt_file is not None:
		print('Restoring model from {}...'.format(ckpt_file))
		saver.restore(sess, ckpt_file)
	for epoch in range(epochs):
		time_train = -time.time()
		index = np.arange(boards.shape[0])
		np.random.shuffle(index)
		losses = []
		regs = []
		for iter in range(batch_num):
			_, l, r, value, prob = sess.run([train_op, loss, reg, v, p], feed_dict={x:boards[index[iter*batch_size:(iter+1)*batch_size]],
																				z:wins[index[iter*batch_size:(iter+1)*batch_size]],
																				pi:ps[index[iter*batch_size:(iter+1)*batch_size]],
																				is_training:True})
			losses.append(l)
			regs.append(r)
			if iter % 1 == 0:
				print("Epoch: {}, Iteration: {}, Time: {}, Loss: {}, Reg: {}".format(epoch, iter, time.time()+time_train, np.mean(np.array(losses)), np.mean(np.array(regs))))
				time_train=-time.time()
				losses = []
				regs = []
			if iter % 20 == 0:
				save_path = "Epoch{}.Iteration{}.ckpt".format(epoch, iter)
				saver.save(sess, result_path + save_path)
