import numpy as np
from math import factorial, exp
from scipy.stats import poisson
from pdb import set_trace as debugger

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
move_cars_cost = 2.0
discount_rate = 0.9

# Precalculated customer appearence probabilities for each location
rent_probs1 = np.array([poisson.pmf(x, rent_mu1) for x in range(l1_places + 1)])
rent_probs2 = np.array([poisson.pmf(x, rent_mu2) for x in range(l2_places + 1)])
# Precalculated rewards for customers count a day
rewards = np.array(range(max(l1_places, l2_places) + 1))
#Pre-calculated probabilities of car returns
ret_probs1 =  np.array([poisson.pmf(x, ret_mu1) for x in range(l1_places + 1)])
ret_probs2 =  np.array([poisson.pmf(x, ret_mu2) for x in range(l2_places + 1)])

# Initialize state values arbitrary
states = np.zeros((l1_places + 1, l2_places + 1))

# Initial policy is not to move anything anywhere at night
policy = np.zeros((l1_places + 1, l2_places + 1), dtype=np.int32)

# Run states value - policy update iterations
for t in range(10):
  print("######Iteration %2d######" % t)
  delta = 100
  # Evaluate strategy
  while delta > 1.0:
    new_states = np.copy(states)
    new_states.fill(1)
    for i in range(l1_places + 1):
      for j in range(l2_places + 1):
        # Nomber of cars to move from first place to the second one overnight
        cars_to_move = policy[i, j]
        new_state_value = 0

        for r1 in range(0, l1_places - i + 1):
          for r2 in range(0, l2_places - j + 1):
            return_prob = ret_probs1[r1] * ret_probs2[r2]

            l1, l2 = i - cars_to_move + r1, j + cars_to_move + r2

            # Probabilities to end up with [l, l-1, l-2,..., 0] cars from given state
            l1_rent_probs1 = np.append(rent_probs1[:l1], [1 - rent_probs1[:l1].sum()])
            l2_rent_probs2 = np.append(rent_probs2[:l2], [1 - rent_probs2[:l2].sum()])
            # Rewards to receive ending up with [l, l-1, l-2, ..., 0] cars from given state
            rewards1 = rent_cost * rewards[:l1 + 1]
            rewards2 = rent_cost * rewards[:l2 + 1]

            # Matrix of transition probabilities
            t_probs = l1_rent_probs1.reshape(l1 + 1, 1) * l2_rent_probs2.reshape(1, l2 + 1)
            # Matrix of rewards
            t_rewards = rewards1.reshape(l1 + 1, 1) + \
                        rewards2.reshape(1, l2 + 1) - \
                        np.abs(cars_to_move) * move_cars_cost

            # Take subset of states available for transition
            t_states = np.flip(states[:l1 + 1, :l2 + 1], axis = (0,1))

            # Calculate new state value according to policy
            new_state_value += return_prob * \
              ((t_rewards * t_probs).sum() + (t_states * t_probs).sum() * discount_rate)

        new_states[i, j] = new_state_value

    delta = np.abs((states - new_states).sum())
    states = new_states

    print("Delta: %4.4f" % delta)

  # Update strategy greedily
  for i in range(l1_places + 1):
    for j in range(l2_places + 1):
      # Search for the best state within 5 cars move
      best_state = states[i, j]
      policy[i, j] = 0
      for k in range(-5, 6):
        if (i - k >= 0) and (i - k < l1_places + 1) and (j + k >= 0) and (j + k < l2_places + 1):
          try_state = states[i - k, j + k]
          if try_state > best_state:
            # If better state value exist - take it as new policy
            best_state = try_state
            policy[i,j] = k

  debugger()
  123
inspect(np.flip(policy, axis = 0))
