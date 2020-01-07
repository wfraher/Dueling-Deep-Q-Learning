#General Atari learner using Dueling Double Deep Q-Learning
#By William Fraher
#Some code in here is by Arthur Juliani, namely the update step and parts of the network architecture. I used his examples to study Q-Learning agents.
#The rest of this is written by me, though it uses a wrapper from OpenAI Baselines to facilitate frame stacking.


import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import gym
from skimage.transform import resize

import baselines
env = baselines.wrap_deepmind(gym.make('PongDeterministic-v4'),frame_stack=True)

class network:
  
  def __init__(self, lr):
    
    #Convolutional DQN.
    
    self.lr = lr
    self.input = tf.placeholder(shape=[None,336*84*1],dtype=tf.float32) #takes a 84x84x3 region as input
    self.reshape = tf.reshape(self.input, shape=[-1,336,84,1])
    self.c1 = layers.conv2d(\
                    self.reshape,
                    num_outputs = 32,
                    kernel_size = [8,8],
                    stride = [4,4],
                    padding = 'valid',
                    biases_initializer = None
                    )
    self.c2 = layers.conv2d(\
                    self.c1,
                    num_outputs=64,
                    kernel_size = [4,4],
                    stride = [2,2],
                    padding = 'valid',
                    biases_initializer = None
                    )
    self.c3 = layers.conv2d(\
                    self.c2,
                    num_outputs = 64,
                    kernel_size = [3,3],
                    stride = [1,1],
                    padding = 'valid',
                    biases_initializer = None
                    )
    self.c4 = layers.conv2d(\
                    self.c3,
                    num_outputs = 512,
                    kernel_size=[7,7],
                    stride=[1,1],
                    padding = 'valid',
                    biases_initializer = None
                    )
    self.aStream,self.vStream = tf.split(self.c4,2,3)
    self.streamA = layers.flatten(self.aStream)
    self.streamV = layers.flatten(self.vStream)
    xavier_init = tf.contrib.layers.xavier_initializer()
    self.AW = tf.Variable(xavier_init([8192,env.action_space.n]))
    self.VW = tf.Variable(xavier_init([8192,1]))
    
    self.Advantage = tf.matmul(self.streamA,self.AW)
    self.Value = tf.matmul(self.streamV,self.VW)
    
    self.qvals = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
    self.predict = tf.argmax(self.qvals,1)
    
    self.target = tf.placeholder(shape=[None], dtype = tf.float32, name='target')

    self.actions = tf.placeholder(shape=[None], dtype = tf.int32, name='actions')
    self.action_onehots = tf.one_hot(self.actions,env.action_space.n, dtype = tf.float32, name='action_onehots')
    
    self.q = tf.reduce_sum(tf.multiply(self.qvals, self.action_onehots, name='computing_qvals'), axis=1, name='suming_qvals')
    self.tderror = tf.square(self.target - self.q)
    self.loss = tf.reduce_mean(self.tderror)
    self.adam = tf.train.AdamOptimizer(learning_rate=self.lr)
    self.step = self.adam.minimize(self.loss)

class experience:
  
  def __init__(self, maxlen=50000):
    self.maxlen = maxlen
    self.data = []
  
  def add(self, new):
    if len(new) + len(self.data) > self.maxlen:
      self.data[0:(len(new) + len(self.data)) - self.maxlen] = []
    self.data.extend(new)
    
  def sample(self,size):
    return np.reshape(np.array(random.sample(self.data,size)),[size,5])
  
  def __len__(self):
    return len(self.data)

def updateTargetGraph(tfVars, tau):
  total_vars = len(tfVars)
  op_holder = []
  for idx,var in enumerate(tfVars[0:total_vars//2]):
    op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
  return op_holder

def updateTarget(op_holder,sess):
  for op in op_holder:
    sess.run(op)

learning_rate = 0.0001
gamma = 0.99
epsilon = 0.99
epsilon_start = 1.0
epsilon_decay = 10000
epsilon_min = 0.1
num_episodes = 100000
pre_exploration_steps = 10000
max_training_steps = 10000
steps_per_update = 4
batchsize = 32
tau = 0.001
maxScore = float("-inf")
load = False

observation_size = 84 * 84 * 1
action_size = env.action_space.n

def preprocess(state,batchsize):
  #preprocess states, with the batchsize equal to the given size
  if batchsize > 1:
    return np.array(np.reshape(state, [batchsize,336*84*1]))
  return np.array(np.reshape(state, [336*84*1]))

tf.reset_default_graph()
onlineQN = network(learning_rate)
targetQN = network(learning_rate) #lr doesn't matter for the target network

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables, tau)

memory = experience(50000)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  
  if load:
    ckpt = tf.train.get_checkpoint_state('./data')
    saver.restore(sess,ckpt.model_checkpoint_path)

  rewards = []
  
  totalSteps = 0
  
  for i in range(num_episodes):
    
    s = env.reset()
    
    d = False
    
    totalRewards = 0
    
    while not d:
      totalSteps += 1
      
      if np.random.rand(1) <= epsilon:
        a = np.random.randint(0,action_size)
      else:        
        feed_dict = {onlineQN.input:[preprocess(s,1)]}
        a = sess.run(onlineQN.predict, feed_dict)[0]
      
      ns, r, d, _ = env.step(a)
      totalRewards += r
      memory.add(np.reshape(np.array([s,a,r,ns,d]),[1,5]))
      
      if totalSteps > pre_exploration_steps:     
        #After initial exploration has been done, we can decrease epsilon as time goes on.
        if epsilon > epsilon_min:
          epsilon_rate = (epsilon_start - epsilon_min) / epsilon_decay
          epsilon = max(epsilon_min, epsilon-epsilon_rate)
        #Similarly, we can begin learning.
        if len(memory) > batchsize:
          if totalSteps % steps_per_update == 0:
            minibatch = memory.sample(batchsize) #add a conditional before this
            #Create the target value
            feed_dict = {onlineQN.input:preprocess(np.vstack(minibatch[:,3]),batchsize)}
            Q1 = sess.run(onlineQN.predict,feed_dict)
            feed_dict = {targetQN.input:preprocess(np.vstack(minibatch[:,3]),batchsize)}
            Q2 = sess.run(targetQN.qvals,feed_dict)
            double_q = Q2[range(batchsize),Q1]
            done_values = -(minibatch[:,4] - 1) #will become 1 if this is not the last step in an episode, 0 otherwise
            target_q = minibatch[:,2] + (gamma * double_q * done_values)
            feed_dict = {onlineQN.input:preprocess(np.vstack(minibatch[:,0]),batchsize), onlineQN.target:target_q, onlineQN.actions:minibatch[:,1]}
            _ = sess.run(onlineQN.step,feed_dict)            
            updateTarget(targetOps, sess)
      
      #Advance the state for the next calculations.
      s = ns
      
      if d:
        break     
    rewards.append(totalRewards)        
        
    if len(rewards) % 10 == 0 and i > 1:
      print(str(totalSteps) + ' ' + str(np.mean(rewards[-10:])) + ' ' + str(epsilon))
      if (np.mean(rewards[-10:])) > maxScore:
        maxScore = np.mean(rewards[-10:])
        saver = tf.train.Saver()
        saver.save(sess,'data/pong.ckpt')
