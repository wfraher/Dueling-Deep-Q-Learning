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
update_freq = 1
update_target_freq = 50


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

def getUpdates(variables):
    onlineW = variables[0:len(variables)//2] #weights of the online Q Network
    targetW = variables[len(variables)//2:len(variables)] #weights of the target network
    operations = []
    for v in range(len(variables)//2):
        operations.append(tf.assign(targetW[v],onlineW[v]))
    return operations

def updateTarget(operations, sess):
    for operation in operations:
        sess.run(operation)

tf.reset_default_graph()

network = QNetwork()
targetNetwork = QNetwork()

variables = tf.trainable_variables()

updates = getUpdates(variables)

total_rewards = []
total_steps = 0
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
	    total_steps += 1
            episode_reward += r
            ns = preprocess(ns, 1)
            memory.append((s,a,r,ns,d))
            s = ns
            if len(memory) >= batchsize and total_steps % update_freq == 0:
                minibatch = np.reshape(np.array(random.sample(memory, batchsize)),[batchsize,5])
                nextVals = sess.run(targetNetwork.predict, feed_dict={targetNetwork.inputs:np.vstack(minibatch[:,3])})
                actions = np.argmax(sess.run(network.predict, feed_dict={network.inputs:np.vstack(minibatch[:,3])}), axis = 1)
                targets = nextVals[range(batchsize),actions]
                target = np.hstack(minibatch[:,2]) + (discount * targets) * (-1 * np.hstack(minibatch[:,4] - 1))
                sess.run(network.step, feed_dict={network.inputs:np.vstack(minibatch[:,0]), network.targets:target, network.actions:minibatch[:,1]})
            if total_steps % update_target_freq == 0:
		updateTarget(updates, sess)
            if d:
                print 'Episode ' + str(i) + ' finished with reward ' + str(episode_reward) + ' with epsilon ' + str(epsilon)
                total_rewards.append(episode_reward)
                break

print 'Elapsed time ' + str(time.time() - now)

plt.plot(total_rewards)
plt.show()
