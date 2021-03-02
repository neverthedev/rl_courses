import numpy as np
from math import factorial, exp
from scipy.stats import poisson
from pdb import set_trace as debugger

np.set_printoptions(linewidth= 100)

# inspect(np.flip(policy, axis = 0))
def inspect(array):
  res = []
  for i in array:
    res.append(', '.join(map(format, i)))
  print("\n".join(res))

def format(number):
  return "{0:3d}".format(number)

l1_places = 20
l2_places = 20

rent_mu1, rent_mu2 = 3, 4
ret_mu1, ret_mu2 = 3, 2
rent_cost = 10
move_car_cost = 2.0
discount_rate = 0.9

# Precalculated customer appearence probabilities for each location
rent_probs1 = np.array([poisson.pmf(x, rent_mu1) for x in range(l1_places + 1)])
rent_probs2 = np.array([poisson.pmf(x, rent_mu2) for x in range(l2_places + 1)])
# Precalculated rewards for customers count a day
rewards = np.array(range(max(l1_places, l2_places) + 1))
#Pre-calculated probabilities of car returns
ret_probs1 =  np.array([poisson.pmf(x, ret_mu1) for x in range(l1_places + 1)])
ret_probs2 =  np.array([poisson.pmf(x, ret_mu2) for x in range(l2_places + 1)])

# Initial policy is not to move anything anywhere at night
policy = np.zeros((l1_places + 1, l2_places + 1), dtype = np.int32)
# Initialize state values arbitrary
states = np.zeros((l1_places + 1, l2_places + 1))

# Run states value - policy update iterations
for t in range(10):
  print("##########    Iteration %2d    ##########" % t)
  delta = 100
  exp_return = 0

  # Evaluate strategy
  while delta > 1.0:
    rent_transions_values = np.zeros(states.shape)

    for i in range(states.shape[0]):
      for j in range(states.shape[1]):

        # Probabilities to end up with [l, l-1, l-2,..., 0] cars from given state
        l1_rent_probs = np.append(rent_probs1[:i], [1 - rent_probs1[:i].sum()])
        l2_rent_probs = np.append(rent_probs2[:j], [1 - rent_probs2[:j].sum()])
        # Rewards to receive ending up with [l, l-1, l-2, ..., 0] cars from given state
        rewards1 = rent_cost * rewards[:i + 1]
        rewards2 = rent_cost * rewards[:j + 1]

        # Matrix of transition probabilities
        t_probs = l1_rent_probs.reshape(i + 1, 1) * l2_rent_probs.reshape(1, j + 1)
        # Matrix of rewards
        t_rewards = rewards1.reshape(i + 1, 1) + \
                    rewards2.reshape(1, j + 1)


        # Take subset of states available for transition
        t_states = np.flip(states[:i + 1, :j + 1], axis = (0,1))

        # Calculate new state value according to policy
        rent_transions_values[i, j] = (t_rewards * t_probs).sum() + \
                                      (t_states * t_probs).sum() * discount_rate

    new_states = np.zeros(states.shape)

    for i in range(states.shape[0]):
      for j in range(states.shape[1]):
        cars_to_move = policy[i, j]
        # Number of cars to move from first place to the second one overnight
        move_cars_cost =  np.abs(cars_to_move) * move_car_cost

        l1, l2 = i - cars_to_move , j + cars_to_move

        l1_ret_probs = ret_probs1[:states.shape[0] - l1 - 1]
        l1_ret_probs = np.append(l1_ret_probs, [1 - l1_ret_probs.sum()])

        l2_ret_probs = ret_probs2[:states.shape[1] - l2 - 1]
        l2_ret_probs = np.append(l2_ret_probs, [1 - l2_ret_probs.sum()])

        ret_probs = l1_ret_probs.reshape(-1, 1) * l2_ret_probs.reshape(1, -1)

        new_states[i, j] = (rent_transions_values[l1:,l2:] * ret_probs).sum()  - move_cars_cost

    delta = np.abs((states - new_states).sum())
    states = new_states

    #print("Delta: %4.4f" % delta)

  print("##########    Expected return: %4.4f    ##########" % \
    (new_states.sum() / (new_states.shape[0] * new_states.shape[1])))

  # Update strategy greedily
  for i in range(states.shape[0]):
    for j in range(states.shape[1]):
      # Search for the best state within 5 cars move
      best_state = states[i, j]
      policy[i, j] = 0
      for k in range(-5, 6):
        if (i - k >= 0) and (i - k < states.shape[0]) and \
           (j + k >= 0) and (j + k < states.shape[1]):

          try_state = states[i - k, j + k]
          if try_state > best_state:
            # If better state value exist - take it as new policy
            best_state = try_state
            policy[i, j] = k

debugger()
inspect(np.flip(policy, axis = 0))

"""
Examle of expected policy 

[[ 5,  5,  5,  4,  3,  3,  2,  2,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  0,  0,  0],
 [ 5,  5,  5,  4,  3,  2,  2,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 5,  5,  5,  4,  3,  2,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 5,  5,  5,  4,  3,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 5,  5,  5,  4,  3,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 5,  5,  5,  4,  3,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 5,  5,  5,  4,  3,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 5,  5,  4,  4,  3,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 5,  5,  4,  3,  3,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 5,  4,  4,  3,  2,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 5,  4,  3,  3,  2,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 4,  4,  3,  2,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 4,  3,  3,  2,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 3,  3,  2,  2,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 2,  2,  2,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
 [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1],
 [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -2, -2, -2, -2, -2],
 [ 0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -2, -2, -2, -2, -2, -3, -3, -3, -3],
 [ 0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -2, -2, -2, -3, -3, -3, -3, -3, -4, -4, -4]],

"""
