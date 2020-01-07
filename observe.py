import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import gym
from gridworld import gameEnv
from skimage.transform import resize
import time

import baselines
env = baselines.wrap_deepmind(gym.make('PongDeterministic-v4'),frame_stack=True)
#env = gym.wrappers.Monitor(env, './recordings', force=True, video_callable=lambda episode_id: True)

class network:
  
  def __init__(self):
    
    #Convolutional DQN. Uses the same architecture as that of the nature paper.
    
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

num_episodes = 100000

observation_size = 84 * 84 * 1
action_size = env.action_space.n

def preprocess(state,batchsize):
  #preprocess states, with the batchsize equal to the given size
  if batchsize > 1:
    return np.array(np.reshape(state, [batchsize,336*84*1]))
  return np.array(np.reshape(state, [336*84*1]))

tf.reset_default_graph()
onlineQN = network()

saver = tf.train.Saver()

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  ckpt = tf.train.get_checkpoint_state('./data')
  saver.restore(sess,ckpt.model_checkpoint_path)
  
  rewards = []
  
  totalSteps = 0
  
  for i in range(num_episodes):
    
    s = env.reset()
    
    d = False
    
    totalRewards = 0
    
    j = 0 #step counter, don't use this for atari, used to terminate episodes
    
    while not d:
      env.render()
      time.sleep(0.01)
      j += 1
      totalSteps += 1
      feed_dict = {onlineQN.input:[preprocess(s,1)]}
      a = sess.run(onlineQN.predict, feed_dict)[0]
      
      ns, r, d, _ = env.step(a)
      totalRewards += r
      
      #Advance the state for the next calculations.
      s = ns
      
      if d:
        break     
    rewards.append(totalRewards)        
        
    if len(rewards) % 10 == 0 and i > 1:
      print(str(totalSteps) + ' ' + str(np.mean(rewards[-10:])) + ' ' + str(epsilon))
      if (np.mean(rewards[-10:])) > maxScore:
        maxScore = np.mean(rewards[-10:])
