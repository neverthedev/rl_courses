import numpy as np
from math import factorial, exp
from scipy.stats import poisson
from pdb import set_trace as debugger

np.set_printoptions(linewidth = 250, precision = 3)

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
state_values = np.zeros((l1_places + 1, l2_places + 1))

def get_transition_values(state_values):
  return_transition_values = np.zeros(state_values.shape)

  for i in range(state_values.shape[0]):
    for j in range(state_values.shape[1]):
      # Probabilities to end up with [l, l+1, l+2,..., l_places] cars after cars returned
      l1_ret_probs = ret_probs1[:state_values.shape[0] - i - 1]
      l1_ret_probs = np.append(l1_ret_probs, [1 - l1_ret_probs.sum()])

      l2_ret_probs = ret_probs2[:state_values.shape[1] - j - 1]
      l2_ret_probs = np.append(l2_ret_probs, [1 - l2_ret_probs.sum()])

      # Matrix of transition probabilities (in each possible state)
      ret_probs = l1_ret_probs.reshape(-1, 1) * l2_ret_probs.reshape(1, -1)

      # State values without cars return factor
      return_transition_values[i, j] = (state_values[i:,j:] * ret_probs).sum() * discount_rate

  rent_transition_values = np.zeros(state_values.shape)

  for i in range(state_values.shape[0]):
    for j in range(state_values.shape[1]):
      # Probabilities to end up with [l, l-1, l-2,..., 0] cars from given state after cars rented
      l1_rent_probs = np.append(rent_probs1[:i], [1 - rent_probs1[:i].sum()])
      l2_rent_probs = np.append(rent_probs2[:j], [1 - rent_probs2[:j].sum()])

      # Rewards to receive ending up with [l, l-1, l-2, ..., 0] cars from given state
      rewards1 = rent_cost * rewards[:i + 1]
      rewards2 = rent_cost * rewards[:j + 1]

      # Matrix of transition probabilities
      rent_probs = l1_rent_probs.reshape(i + 1, 1) * l2_rent_probs.reshape(1, j + 1)

      # Take subset of states available for transition
      transition_state_values = np.flip(return_transition_values[:i + 1, :j + 1], axis = (0,1))

      # Matrix of rewards
      transition_rewards = rewards1.reshape(i + 1, 1) + rewards2.reshape(1, j + 1)

      # Calculate new state values before cars rent factor
      rent_transition_values[i, j] = (transition_rewards * rent_probs).sum() + \
                                     (transition_state_values * rent_probs).sum()

  return rent_transition_values

def state_under_policy_values(state_values, policy):
  transition_values = get_transition_values(state_values)
  new_state_values = np.zeros(state_values.shape)

  for i in range(state_values.shape[0]):
    for j in range(state_values.shape[1]):
      cars_to_move = policy[i, j]
      # Number of cars to move from first place to the second one overnight
      move_cars_cost =  np.abs(cars_to_move) * move_car_cost

      # Resulting state
      l1, l2 = i - cars_to_move , j + cars_to_move

      new_state_values[i, j] = transition_values[l1, l2] - move_cars_cost

  return new_state_values

def greedy_policy(state_values):
  transition_values = get_transition_values(state_values)
  policy = np.zeros(state_values.shape, dtype = np.int32)

  for i in range(policy.shape[0]):
    for j in range(policy.shape[1]):
      # Search for the best state within 5 cars move
      best_state = transition_values[i, j]
      policy[i, j] = 0
      for k in range(-5, 6):
        if (i - k >= 0) and (i - k < policy.shape[0]) and \
           (j + k >= 0) and (j + k < policy.shape[1]):

          try_state = transition_values[i - k, j + k] - np.abs(k) * move_car_cost
          if try_state >= best_state:
            # If better state value exist - take it as new policy
            best_state = try_state
            policy[i, j] = k

  return policy

# Run states values - policy update iterations
for t in range(100):
  print("##########    Iteration %2d    ##########" % (t + 1))
  delta = 100
  exp_return = 0

  # Evaluate strategy
  while delta > 0.01: # That's quite neat
    new_state_values = state_under_policy_values(state_values, policy)
    delta = np.abs((state_values - new_state_values).sum())
    state_values = new_state_values

    #print("Delta: %4.4f" % delta)

  # Update strategy greedily
  new_policy = greedy_policy(state_values)

  # Stop iterating if policy is stable
  if np.abs((policy - new_policy).sum()) == 0: break

  policy = new_policy

print('Completed in %1d iterations. Final policy:' % (t + 1))
print(np.flip(policy, axis = 0))
print('State values:')
print(np.flip(state_values, axis = 0))

"""
Resulting policy

[[ 5  5  5  5  4  4  3  3  3  3  2  2  2  2  2  1  1  1  0  0  0]
 [ 5  5  5  4  4  3  3  2  2  2  2  1  1  1  1  1  0  0  0  0  0]
 [ 5  5  5  4  3  3  2  2  1  1  1  1  0  0  0  0  0  0  0  0  0]
 [ 5  5  5  4  3  2  2  1  1  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  5  4  3  2  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  5  4  3  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  4  4  3  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  4  3  3  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  5  4  3  2  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 5  4  4  3  2  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 4  4  3  3  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 4  3  3  2  2  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 3  3  2  2  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 3  2  2  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 2  2  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 1  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 -1 -1]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 -1 -1 -1 -1 -1 -2]
 [ 0  0  0  0  0  0  0  0  0  0  0 -1 -1 -1 -1 -1 -2 -2 -2 -2 -2]
 [ 0  0  0  0  0  0  0  0  0 -1 -1 -1 -2 -2 -2 -2 -2 -3 -3 -3 -3]
 [ 0  0  0  0  0  0  0  0 -1 -1 -2 -2 -2 -3 -3 -3 -3 -3 -4 -4 -4]]
"""
