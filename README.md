# Dueling-Deep-Q-Learning
Dueling Deep Q-Learning implementation for Atari 2600 games. This was from April 2018.
This was done as a school project for an AI class I took in the spring semester of 2018 at Central Connecticut State University.

This project contains a dueling deep q-learning agent as well as a vanilla deep q-learning agent without the dueling architecture or double deep q-learning enhancements.
To train the dueling deep q-learning agent, run duelingddqn.py. This will generate weights in the data folder and produce a graph.
You can watch the agent play the game by running observe.py, though this requires weights to be in the data folder.
Additionally, running dqn_atari_target.py trains a deep q-learning agent using a target network and produces a graph, which is included for comparison purposes. Note that observe.py will only accept weights from the dueling deep q-learning agent.
Some of Arthur Juliani's code is in duelingddqn.py, as his reinforcement learning tutorials helped me greatly. I recommend checking his tutorials out if you want to understand more reinforcement learning techniques.
The agents are coded for an environment in OpenAI Gym called PongDeterministic-v4, if you want to train it on a different game change it to something like BreakoutDeterministic-v4 for the game breakout. So replace Pong or Breakout with the desired game in the env.make statements for duelingddqn.py, dqn_atari_target.py, and observe.py.
This could be done with command line arguments but for the interest of time I didn't add them.

This required me to research a topic independently and to present it, I chose reinforcement learning as I had a command of neural networks at the time and thought it would be fun.
The rl_presentation2.pdf and rl_presentation2.odp files are the slides for presentation I gave.
William Fraher - Reinforcement Learning.pdf is a paper I wrote describing reinforcement learning and how Deep Q-Learning works, as well as how other algorithms work and compare.
