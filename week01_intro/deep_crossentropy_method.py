# Extend your CEM implementation with neural networks!
# Train a multi-layer neural network to solve simple continuous state space games

import gym
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pdb import set_trace as debugger

# Open and initialize CartPole problem

env = gym.make("CartPole-v0").env
observation = env.reset()

n_actions = env.action_space.n
state_dim = env.observation_space.shape[0]

#plt.imshow(env.render("rgb_array"))
print("State vector dimension: ", state_dim)
print("Number of actions: ", n_actions)

# For this assignment we'll utilize the simplified neural network implementation from
# Scikit-learn. Here's what you'll need:
#   agent.partial_fit(states, actions) - make a single training pass over the data.
#                                        Maximize the probabilitity of :actions: from :states:
#   agent.predict_proba(states) - predict probabilities of all actions, a matrix of
#                                 shape [len(states), n_actions]

from sklearn.neural_network import MLPClassifier

agent = MLPClassifier(hidden_layer_sizes=(20, 20), activation='tanh')

# initialize agent to the dimension of state space and number of actionscurrent_state = env.reset()
agent.partial_fit([observation] * n_actions, range(n_actions), classes=range(n_actions))

def generate_session(env, agent, t_max=1000):
  #  Play a single game using agent neural network.
  #  Terminate when game finishes or after :t_max: steps
  states, actions = [], []
  total_reward = 0

  state = env.reset()

  for t in range(t_max):
    probs = agent.predict_proba([state])[0]

    assert probs.shape == (env.action_space.n,), "make sure probabilities are a vector (hint: np.reshape)"

    action = np.random.choice(n_actions, p=probs)

    new_s, r, done, info = env.step(action)

    # record sessions like you did before
    states.append(state)
    actions.append(action)
    total_reward += r

    state = new_s

    if done: break

  return states, actions, total_reward

def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
  """
  Select states and actions from games that have rewards >= percentile
  :param states_batch: list of lists of states, states_batch[session_i][t]
  :param actions_batch: list of lists of actions, actions_batch[session_i][t]
  :param rewards_batch: list of rewards, rewards_batch[session_i]

  :returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions

  Please return elite states and actions in their original order
  [i.e. sorted by session number and timestep within session]

  If you are confused, see examples below. Please don't assume that states are integers
  (they will become different later).
  """
  reward_threshold = np.percentile(rewards_batch, percentile)

  elite_states, elite_actions = [], []

  for index, reward in enumerate(rewards_batch):
    if reward >= reward_threshold:
        elite_states.extend(states_batch[index])
        elite_actions.extend(actions_batch[index])

  return elite_states, elite_actions


from IPython.display import clear_output

log = []
n_sessions = 100
percentile = 70

plt.ion()

graph = plt.figure(figsize=[8, 4])
fig1 = graph.add_subplot(1, 2, 1)
fig2 = graph.add_subplot(1, 2, 2)

def show_progress(rewards_batch, log, percentile, reward_range=[-990, +10]):
  # https://stackoverflow.com/questions/4098131/how-to-update-a-plot-in-matplotlib
  # A convenience function that displays training progress.
  # No cool math here, just charts.
  mean_reward = np.mean(rewards_batch)
  threshold = np.percentile(rewards_batch, percentile)
  log.append([mean_reward, threshold])

  rewards, thresholds = list(zip(*log))

  fig1.clear()
  fig1.plot(list(zip(*log))[0], label='Mean rewards')
  fig1.plot(list(zip(*log))[1], label='Reward thresholds')
  fig1.legend()
  fig1.grid()

  fig2.clear()
  fig2.hist(rewards_batch, range=reward_range)
  fig2.vlines([np.percentile(rewards_batch, percentile)],
               [0], [100], label="percentile", color='red')
  fig2.legend()
  fig2.grid()

  graph.canvas.draw()
  graph.canvas.flush_events()

  #clear_output(True)
  print("mean reward = %.3f, threshold=%.3f" % (mean_reward, threshold))

for i in range(100):
  # generate new sessions
  sessions = [ generate_session(env, agent, t_max=200) for _ in range(n_sessions) ]

  print('States length: ', len(sessions[0][1]))

  states_batch, actions_batch, rewards_batch = map(np.array, zip(*sessions))

  elite_states, elite_actions = select_elites(
    states_batch, actions_batch, rewards_batch, percentile=percentile
  )

  agent.partial_fit(elite_states, elite_actions)

  show_progress(rewards_batch, log, percentile, reward_range=[0, np.max(rewards_batch)])

  if np.mean(rewards_batch) > 199.5:
    print("You Win! You may stop training now via KeyboardInterrupt.")
    break

# Record sessions

import gym.wrappers

with gym.wrappers.Monitor(gym.make("CartPole-v0"), directory="videos", force=True) as env_monitor:
    sessions = [generate_session(env_monitor, agent) for _ in range(100)]

