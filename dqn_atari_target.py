#By William Fraher

import gym
import tensorflow as tf
import numpy as np
import random
from matplotlib import pyplot as plt
from collections import deque
import time
import baselines
from skimage.transform import resize

env = gym.make('PongDeterministic-v4')
env = baselines.wrap_deepmind(env, frame_stack=True)

lr = 0.00025
discount = 0.99
hidden_size = 512
pre_train_steps = 10000
batchsize = 32

disp_freq = 10

capacity = 50000
epsilon = 0.99
epsilon_start = 0.99
epsilon_decay = 10000
epsilon_min = 0.1
update_freq = 4
update_target_freq = 1000

observation_size = 336 * 84 * 1
action_size = env.action_space.n

class QNetwork:

    def __init__(self):
        self.inputs = tf.placeholder(shape = [None,observation_size], dtype = tf.float32)
        self.reshape = tf.reshape(self.inputs, shape=[-1,336,84,1]) 
        xavier = tf.contrib.layers.xavier_initializer()
        self.conv1 = tf.contrib.layers.conv2d(\
            self.reshape,
            num_outputs = 32,
            kernel_size = [8,8],
            stride = [4,4],
            padding = 'valid',
            biases_initializer = xavier)
        self.conv2 = tf.contrib.layers.conv2d(\
            self.conv1,
            num_outputs = 64,
            kernel_size = [4,4],
            stride = [2,2],
            padding = 'valid',
            biases_initializer = xavier)
        self.conv3 = tf.contrib.layers.conv2d(\
            self.conv2,
            num_outputs = 64,
            kernel_size = [3,3],
            stride=[1,1],
            padding = 'valid',
            biases_initializer = xavier)
        self.flattened = tf.layers.flatten(self.conv3)
        self.denseWeights = tf.Variable(xavier([17024,512]))
        self.denseBias = tf.Variable(xavier([512]))
        self.dense = tf.nn.relu(tf.matmul(self.flattened, self.denseWeights) + self.denseBias)
        self.predictWeights = tf.Variable(xavier([512,action_size]))
        self.predictBias = tf.Variable(xavier([action_size]))
        self.predict = tf.matmul(self.dense, self.predictWeights) + self.predictBias

        self.targets = tf.placeholder(shape = [None], dtype = tf.float32)
        self.actions = tf.placeholder(shape = [None], dtype = tf.uint8)
        self.actions_onehot = tf.one_hot(self.actions,action_size, dtype = tf.float32)
        self.actionValues = tf.reduce_sum(tf.multiply(self.predict, self.actions_onehot), axis = 1)
        
        self.error = self.targets - self.actionValues
        self.loss = tf.reduce_mean(tf.square(self.error))
        self.Adam = tf.train.AdamOptimizer(learning_rate=lr)
        self.step = self.Adam.minimize(self.loss)

def preprocess(state,batchsize):
  #preprocess states, with the batchsize equal to the given size
  return np.array(np.reshape(state, [batchsize,336*84*1]))
  
class experience:

    def __init__(self):
        self.data = []

    def add(self,new):
        self.data.append(new)
        if len(self.data) > capacity:
            diff = len(self.data) - capacity
            self.data[0:diff] = []

    def sample(self):
        return np.array(random.sample(self.data,batchsize))

    def __len__(self):
        return len(self.data)

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
  

memory = experience()

tf.reset_default_graph()
network = QNetwork()
targetNetwork = QNetwork()

variables = tf.trainable_variables()

updates = getUpdates(variables)

total_rewards = []

now = time.time()

total_steps = 0

i = 0

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    while total_steps < 10000000:
        s = env.reset()
        d = False
        episode_reward = 0
        while not d:
            if epsilon < np.random.rand() and total_steps > pre_train_steps:
                a = np.argmax(sess.run(network.predict, feed_dict={network.inputs:preprocess(s,1)}))
            else:
                a = np.random.randint(0,action_size)
            if total_steps > pre_train_steps:
                epsilon_rate = (epsilon_start - epsilon_min) / epsilon_decay
                epsilon = max(epsilon_min, epsilon - epsilon_rate)
            ns, r, d, _ = env.step(a)
            total_steps += 1
            episode_reward += r
            memory.add((s,a,r,ns,d))
            s = ns
            if len(memory) >= batchsize and total_steps % update_freq == 0:
                minibatch = memory.sample()
                nextVal = sess.run(targetNetwork.predict, feed_dict={targetNetwork.inputs:preprocess(np.vstack(minibatch[:,3]),batchsize)})
                nextVal = np.max(nextVal, axis = 1)
                target = np.hstack(minibatch[:,2]) + (discount * nextVal) * (-1 * np.hstack(minibatch[:,4] - 1))
                sess.run(network.step, feed_dict={network.inputs:preprocess(np.vstack(minibatch[:,0]),batchsize), network.targets:target, network.actions:minibatch[:,1]})
            if total_steps % update_target_freq == 0:
                updateTarget(updates,sess)
            if total_steps % 1e4 == 0:
                saver = tf.train.Saver()
                saver.save(sess,'data/Pong.ckpt')
            if d:
                total_rewards.append(episode_reward)
                break
        if i % disp_freq == 0:
            print 'Episode ' + str(i) + ' avg reward, past ' + str(disp_freq) + ' episodes ' + str(np.mean(total_rewards[-disp_freq:])) + ' with epsilon ' + str(epsilon) + ' at training step ' + str(total_steps)            
        i += 1

print 'Elapsed time ' + str(time.time() - now)

plt.plot(total_rewards[0::disp_freq])
plt.show()
print np.mean(total_rewards)
