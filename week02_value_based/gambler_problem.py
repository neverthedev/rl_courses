import sys
import numpy as np
from os import path
from pdb import set_trace as debugger

np.set_printoptions(linewidth = 250, precision = 3)

class GamblerProblem:
  def __init__(self):
    self._reward = 0
    self._targetBudget = 100
    self._finalRewards = []
    self.setInitialState()

  def makeAction(self):
    if self.completed:
      self._currentEpisodeNumber += 1
      self._finalRewards.append(self._reward)
      if self._currentEpisodeNumber < self._episodesLimit:
        self.solutioner.completed(self._reward)
        self.restart()

      return

    bet = max(min(self.solutioner.getNextAction(self._budget, self._reward), self._budget), 0)

    #print(self._budget, bet)

    if np.random.choice(2, 1, p=[0.6, 0.4])[0] == 1:
      self._budget = min(self._budget + bet, self._targetBudget)
      if self._budget >= self._targetBudget: self._reward = 1
    else:
      self._budget = max(self._budget - bet, 0)


  def setInitialState(self):
    self._budget = 10

  def restart(self):
    self._reward = 0
    self.setInitialState()

  @property
  def completed(self):
    return self._budget == 0 or self._budget == self._targetBudget

  def run(self, solutioner, episodesLimit = 1):
    self.solutioner = solutioner
    self._episodesLimit = episodesLimit
    self._currentEpisodeNumber = 0
    if self.solutioner: self.solutioner.bootstrap()

    while not (self.completed and (self._currentEpisodeNumber >= self._episodesLimit)):
      self.makeAction()

    print('Final rewards are: ', self._finalRewards)


class DPSolutioner:
  def __init__(self):
    self._targetBudget = 100
    self._actionValues = np.zeros((self._targetBudget + 1, self._targetBudget - 1))

  @property
  def _policy(self):
    res = []
    for x in self._actionValues:
      action = 0
      for i, y in enumerate(x):
        if round(y, 10) > round(x[action],10): action = i
      res.append(action + 1)
    return res # np.argmax(self._actionValues, 1) + 1

  def valueIteration(self):
    stateValues = np.max(self._actionValues, 1)
    stateValues[self._targetBudget] = 1 # There is only one rewarding state

    states = np.arange(0, self._targetBudget + 1).reshape(-1, 1)
    actions = np.arange(1, self._targetBudget).reshape(1, -1)

    winningStates = states + actions
    loosingStates = states - actions

    winningStates[loosingStates < 0] = 0 # when gambler bets more than he has
    winningStates[winningStates > self._targetBudget] = self._targetBudget
    loosingStates[loosingStates < 0] = 0

    stateValuesDict = { i: v for i, v in enumerate(stateValues) }

    actionValues = 0.6 * np.vectorize(stateValuesDict.get)(loosingStates) + \
                   0.4 * np.vectorize(stateValuesDict.get)(winningStates)

    actionValues[-1,:] = 1 # Final state's actions value is always 1

    return actionValues

  def bootstrap(self):
    delta, t = 100., 0

    # Evaluate strategy
    while delta > 0.0001: # That's quite neat
      print("##########    Iteration %2d    ##########, Delta: %3.3f" % (t + 1, delta))
      newActionValues = self.valueIteration()

      delta = np.abs((self._actionValues - newActionValues).sum())
      self._actionValues = newActionValues

      t += 1

    return t

  def getNextAction(self, budget, reward):
    return self._policy[budget]

  def completed(self, reward):
    pass

game = GamblerProblem()
game.run(solutioner = DPSolutioner(), episodesLimit = 10)

