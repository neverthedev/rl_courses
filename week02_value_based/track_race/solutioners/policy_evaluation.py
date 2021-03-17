import numpy as np
import sys, os
from pdb import set_trace as debugger

sys.path.append(os.getcwd())

import random as rand

class PolicyEvaluation:

  def __init__(self, filename):
    self.filename = filename
    self._speedLimit = 5
    self._speedsCount = self._speedLimit * 2 + 1
    self._dropOffPenalty = 100
    self._transitionCost = 1

    # Load track schema
    self.schema = np.flip(np.loadtxt(self.filename + '.dat', dtype=np.int), axis=0)

    # Initialize state values function with arbitrary values
    self.stateValues = np.zeros(
      (self.schema.shape[0], self.schema.shape[1],  self._speedsCount ** 2)
    )
    # Generate full set of actions
    self._actionsMatrix = self._getActionsMatrix()


  def bootstrap(self):
    policyFileName = self.filename + '.dp.policy.npy'
    if self.loadFromFile(policyFileName):
      self.policy = np.zeros((*self.stateValues.shape, 2), dtype=np.ndarray)
      self.valueIteration(self.policy)
    else:
      self.findOptimalPolicy()
      self.saveToFile(policyFileName)


  def saveToFile(self, fileName):
    np.save(fileName, self.stateValues)


  def loadFromFile(self, fileName):
    try:
      self.stateValues = np.load(fileName)
      return True
    except FileNotFoundError as error:
      print("Can't load state-value function from file. Starting itarations...")
      return False


  def _getActionsMatrix(self):
    incs = [-1, 0, 1]
    return np.array([[ [i, j] for j in incs ] for i in incs])


  def valueIteration(self, policy = None):
    newStateValues = np.zeros(self.stateValues.shape)
    newStateValues[:, :, :] = - np.inf
    dropOffActionValue = self._dropOffActionValue()

    for k in range(self.stateValues.shape[2]):

      if policy is not None: policy[:, :, k] = np.array([0,0])

      # Go on to calculate state value of points on the track
      for action in self._actionsMatrix.reshape(-1, 2):
        currentSpeed = self._decodeSpeed(k)
        vSpeed, hSpeed = action + currentSpeed

        if (vSpeed in range(-self._speedLimit, self._speedLimit + 1)) and \
           (hSpeed in range(-self._speedLimit, self._speedLimit + 1)):

          newSpeed = self._encodeSpeed([vSpeed, hSpeed])

          speedStateValues = self.stateValues[:, :, newSpeed]

          newSpeedStateValues = self._shiftArr(speedStateValues, - vSpeed, - hSpeed, fillWith = 0)

          # Take transition cast for all values
          newSpeedStateValues -= self._transitionCost

          # Find out which transitions cross borders
          borderCrossesMatrix = self._borderCrossesMatrix(int(vSpeed), int(hSpeed))

          # Set drop-off value for each state where car hits border during transition
          newSpeedStateValues[borderCrossesMatrix] = dropOffActionValue

          # Set zero value for off track starting points (as long as they are unreachable anyway)
          newSpeedStateValues[self.schema == 0] = 0

          # Regardles of the speed the value of the terminal state is always zero
          newSpeedStateValues[self.schema == 2] = 0

          # Update policy for current speed...
          policy[newSpeedStateValues > newStateValues[:, :, k], k] = action

          # Update state values for current speed
          newStateValues[:, :, k] = np.maximum(newStateValues[:, :, k], newSpeedStateValues)

    return newStateValues


  def _borderCrossesMatrix(self, dv, dh):
    res = np.zeros((self.stateValues.shape[0], self.stateValues.shape[1]), dtype=np.bool)

    if dv != 0:
      step = 1 if dv >= 0 else -1

      for v in range(step, dv + step, step):
        h = round((dh / dv) * v)
        shPad = self._shiftArr(self.schema, -v, -h, fillWith = 0)
        res = np.logical_or(res, (shPad == 0))

    if dh != 0:
      step = 1 if dh >= 0 else -1

      for h in range(step, dh + step, step):
        v = round((dv / dh) * h)
        shPad = self._shiftArr(self.schema, -v, -h, fillWith = 0)
        res = np.logical_or(res, (shPad == 0))

    return res


  def _shiftArr(self, arr, dv, dh, fillWith = np.nan):
    result = np.copy(arr)

    if dh > 0:
        result[:, dh:] = result[:, :-dh]
        result[:, :dh] = fillWith
    elif dh < 0:
        result[:, :dh] = result[:, -dh:]
        result[:, dh:] = fillWith

    if dv > 0:
        result[dv:, :] = result[:-dv, :]
        result[:dv, :] = fillWith
    elif dv < 0:
        result[:dv, :] = result[-dv:, :]
        result[dv:, :] = fillWith

    return result


  def _encodeSpeed(self, speedVector):
    return (speedVector[0] + self._speedLimit) * self._speedsCount  + \
           (speedVector[1] + self._speedLimit)


  def _decodeSpeed(self, speed):
    vSpeed = speed // self._speedsCount - self._speedLimit
    hSpeed = speed % self._speedsCount - self._speedLimit

    return np.array([vSpeed, hSpeed])


  # This actually doesn't depend on current state or value. Agent always returns
  #  in some of initial state with equal probability for each
  #  (optionally, with drop-off penalty)
  def _dropOffActionValue(self):
    initialStates = np.array(np.where(self.schema[0] == 1)[0])

    actionValue = 0.0

    for hPos in initialStates:
      actionValue += self.stateValues[0, hPos, self._encodeSpeed([0, 0])]

    return actionValue / len(initialStates) - self._dropOffPenalty


  def findOptimalPolicy(self):
    delta, t  = 100, 0
    print('######    Iteration 0   ######')
    while delta > 1.:
      self.policy = np.zeros((*self.stateValues.shape, 2), dtype=np.ndarray)
      newStateValues = self.valueIteration(self.policy)

      delta = np.abs((self.stateValues - newStateValues).sum())
      self.stateValues = newStateValues
      t += 1
      print('######    Iteration %2d, Delta: %3.3f    ######' % (t, delta))


  def getNextAction(self, vPos, hPos, vSpeed, hSpeed):
    if not hasattr(self, 'policy'): self.bootstrap()

    return self.policy[vPos, hPos, self._encodeSpeed([vSpeed, hSpeed])]

# Example of usage
# policy = PolicyEvaluation('traces/trace_cut')
# policy.bootstrap()
# print(policy.getNextAction(0, 5, 0, 0))
