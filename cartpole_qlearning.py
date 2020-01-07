import gym
import tensorflow as tf
import numpy as np
import random
from matplotlib import pyplot as plt
from collections import deque
import time

env = gym.make('CartPole-v1')

lr = 0.0005
max_steps = 500 #maximum for cartpole_v1
iterations = 1000
discount = 0.95
hidden_size = 32
batchsize = 32

memory = deque(maxlen=2000)
epsilon = 0.99
epsilon_start = 0.99
epsilon_decay = 1000
epsilon_min = 0.1

observation_size = env.observation_space.shape[0]
action_size = env.action_space.n

class QNetwork:

    def __init__(self):
        self.inputs = tf.placeholder(shape = [None,observation_size], dtype = tf.float32)
        xavier = tf.contrib.layers.xavier_initializer()
        self.W1 = tf.Variable(xavier([observation_size,hidden_size]))
        self.b1 = tf.Variable(xavier([hidden_size]))
        self.dense = tf.nn.relu(tf.matmul(self.inputs,self.W1) + self.b1)
        self.W2 = tf.Variable(xavier([hidden_size,action_size]))
        self.b2 = tf.Variable(xavier([action_size]))
        self.predict = tf.matmul(self.dense, self.W2) + self.b2

        self.targets = tf.placeholder(shape = [None], dtype = tf.float32)
        self.actions = tf.placeholder(shape = [None], dtype = tf.uint8)
        self.actions_onehot = tf.one_hot(self.actions,action_size, dtype = tf.float32)
        self.actionValues = tf.reduce_sum(tf.multiply(self.predict, self.actions_onehot), axis = 1)
        
        self.error = tf.square(self.targets - self.actionValues)
        self.loss = tf.reduce_mean(self.error)
        self.Adam = tf.train.AdamOptimizer(learning_rate=lr)
        self.step = self.Adam.minimize(self.loss)

def preprocess(states, size):
    return np.reshape(states, [size,observation_size])

tf.reset_default_graph()
network = QNetwork()
total_rewards = []

now = time.time()

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for i in range(iterations):
        s = preprocess(env.reset(),1)
        d = False
        episode_reward = 0
        for t in range(max_steps):
            if epsilon < np.random.rand():
                a = np.argmax(sess.run(network.predict, feed_dict={network.inputs:s}))
            else:
                a = np.random.randint(0,action_size)
                epsilon_rate = (epsilon_start - epsilon_min) / epsilon_decay
                epsilon = max(epsilon_min, epsilon - epsilon_rate)
            ns, r, d, _ = env.step(a)
            episode_reward += r
            ns = preprocess(ns, 1)
            memory.append((s,a,r,ns,d))
            s = ns
            if len(memory) >= batchsize:
                minibatch = np.reshape(np.array(random.sample(memory, batchsize)),[batchsize,5])
                nextVal = sess.run(network.predict, feed_dict={network.inputs:np.vstack(minibatch[:,3])})
                nextVal = np.max(nextVal, axis = 1)
                target = np.hstack(minibatch[:,2]) + (discount * nextVal) * (-1 * np.hstack(minibatch[:,4] - 1))
                sess.run(network.step, feed_dict={network.inputs:np.vstack(minibatch[:,0]), network.targets:target, network.actions:minibatch[:,1]})
            if d:
                print 'Episode ' + str(i) + ' finished with reward ' + str(episode_reward) + ' with epsilon ' + str(epsilon)
                total_rewards.append(episode_reward)
                break

print 'Elapsed time ' + str(time.time() - now)

plt.plot(total_rewards)
plt.show()
