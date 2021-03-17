import numpy as np
import sys, os
import pdb

sys.path.append(os.getcwd())

from race import Race
import random as rand

class TD0Policy:

  def __init__(self):
    self.traceFileName = os.getcwd() + '/traces/trace_cut.dat'
    self.scheme = np.loadtxt(self.traceFileName, dtype=np.int)
    self.stateValues = np.zeros((self.scheme.shape[0], self.scheme.shape[1], 121))
    self.epsilon = 0.2

  @property
  def height(self): # Rows count
    return self.scheme.shape[0]

  @property
  def width(self): # Columns count 
    return self.scheme.shape[1]

  @property
  def race(self):
    if not hasattr(self, '_race'):
      self._race = Race(traceFile = self.traceFileName)
    return self._race
  
  # Generate experience using epsolon-greedy policy derived from state values
  #
  # @length - the number of steps to make according to policy being evaluated
  def generateExperience(self, length, visualize=False):
    rand.seed(12)
    # Because here we have model of the system let's use Exploring Starts
    speed = rand.randint(0, 11) * 11 + rand.randint(0, 11) # first is vertical speed(y), second - horizontal(x)
    position = rand.randint(0, self.height - 1), rand.randint(0, self.width - 1), speed

    self.race.setInitialPosition(position)

    self._stepsLeft = length
    self._visualize = visualize
    self._expStates = [self.race.state]
    self._expRewards = []

    nextStates = self.race.nextStates()

    if self._visualize:
      self.race.gamefield.subscribeEvent('space', self.makeAction)
      self.race.gamefield.subscribeEvent('exit', self.exit)
      self.race.run()
    else:
      while self._stepsLeft > 0:
        self.makeAction()

  def makeAction(self, visualize=False):
    # Speed can only be changed by 1 at each axis at a time
    speedDeltaX, speedDeltaY = rand.randint(0, 2) - 1, rand.randint(0, 2) - 1
    self.race.makeAction((speedDeltaX, speedDeltaY))
    reward = self.race.nextState()

    self._expRewards.append(reward)
    self._expStates.append(self.race.state)

    self._stepsLeft -= 1
    if self._visualize:
      if self._stepsLeft > 0:
        self.race.displayCurrentGameState()
      else:
        self.race.gamefield.quit()
  
  def exit(self):
    self.race.gamefield.quit(); 
    sys.exit();

  # Evaluate policy given experience 
  #
  # @experience - sequence of states, actions and rewards received 
  #               while following policy being evaluated
  def evaluatePolicy(self, experience):
    # Evaluate epsilon-greedy policy with respect to state values function
    alpha = 0.1
    for idx, reward in enumerate(self._expRewards):
      prev_state_value = self.stateValues[self._expStates[idx + 1]]
      next_state_value = self.stateValues[self._expStates[idx + 1]]
      self.stateValues[self._expStates[idx]] += alpha * (reward + next_state_value - prev_state_value)
      
td0 = TD0Policy()
experience = td0.generateExperience(1000, visualize=False)
td0.evaluatePolicy(experience)
