import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

import io
import sys
import csv

# Path environment changed to make things work properly
# export DYLD_FALLBACK_LIBRARY_PATH=$DYLD_FALLBACK_LIBRARY_PATH:/usr/lib


# Get the environment and extract the number of actions.
ENV_NAME = 'LunarLander-v2'
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
#print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=300000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# After training is done, we save the final weights.
dqn.load_weights('dqn_{}_weights_model3.h5f'.format(ENV_NAME))

# Redirect stdout to capture test results
old_stdout = sys.stdout
sys.stdout = mystdout = io.StringIO()

# Evaluate our algorithm for a few episodes.
dqn.test(env, nb_episodes=200, visualize=False)

# Reset stdout
sys.stdout = old_stdout

results_text = mystdout.getvalue()

# Print results text
print("results")
print(results_text)

# Extact a rewards list from the results
total_rewards = list()
for idx, line in enumerate(results_text.split('\n')):
    if idx > 0 and len(line) > 1:
        reward = float(line.split(':')[2].split(',')[0].strip())
        total_rewards.append(reward)

# Print rewards and average	
print("total rewards", total_rewards)
print("average total reward", np.mean(total_rewards))

# Write total rewards to file
f = open("lunarlander_rl_rewards.csv",'w')
wr = csv.writer(f)
for r in total_rewards:
     wr.writerow([r,])
f.close()