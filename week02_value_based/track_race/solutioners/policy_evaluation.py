import numpy as np
import sys, os
from pdb import set_trace as debugger


sys.path.append(os.getcwd())

import random as rand

class PolicyEvaluation:

  def __init__(self):
    self.filename = '/traces/trace_cut'
    self._speedLimit = 5
    self._speedsCount = self._speedLimit * 2 + 1
    self._dropOffPenalty = 100
    self._transitionCost = 1

    # Load track schema
    self.traceFileName = os.getcwd() + self.filename + '.dat'
    self.schema = np.flip(np.loadtxt(self.traceFileName, dtype=np.int), axis=0)

    # Initialize state values function with arbitrary values
    self.stateValues = np.zeros(
      (self.schema.shape[0], self.schema.shape[1],  self._speedsCount ** 2)
    )
    # Generate full set of actions
    self._actionsMatrix = self._getActionsMatrix()

  def bootstrap(self):
    policyFileName = self.filename + 'dp.policy.npy'
    if self.loadFromFile(policyFileName):
      self.policy = np.zeros(self.stateValues.shape, dtype=np.ndarray)
      self.valueIteration(self.policy)
    else:
      self.findOptimalPolicy()
      self.saveToFile(policyFileName)

  def saveToFile(self, fileName):
    fullFilePath = os.getcwd() + fileName
    np.save(fullFilePath, self.stateValues)

  def loadFromFile(self, fileName):
    fullFilePath = os.getcwd() + fileName
    try:
      self.stateValues = np.load(fullFilePath)
      return True
    except FileNotFoundError as error:
      print("Can't load state-value function from file. Starting itarations...")
      return False

  def _getActionsMatrix(self):
    incs = [-1, 0, 1]
    return np.array([[ [i, j] for j in incs ] for i in incs])


  def valueIteration(self, policy = None):
    newStateValues = np.zeros(self.stateValues.shape)
    dropOffActionValue = self._dropOffActionValue()

    for i in range(self.stateValues.shape[0]):
      for j in range(self.stateValues.shape[1]):
        # There is actually no state at the side of the track. So no need to calculate
        #   the next states range for them
        if self.schema[i, j] == 0: continue

        for k in range(self.stateValues.shape[2]):
          # Regardles of the speed the value of the terminal state is always zero
          if self.schema[i, j] == 2:
            newStateValues[i, j, k] = 0
            continue

          # Go on to calculate state value of points on the track
          newStateValues[i, j, k] = - np.inf

          if policy is not None: policy[i, j, k] = np.array([0,0])

          speedsMatrix = self._actionsMatrix + self._decodeSpeed(k)
          for action in self._actionsMatrix.reshape(-1, 2):
            currentSpeed = self._decodeSpeed(k)
            vSpeed, hSpeed = action + currentSpeed

            # Check if the action is possible (speed is not exceeded)
            if (vSpeed >= -self._speedLimit) and (vSpeed <= self._speedLimit) and \
               (hSpeed >= -self._speedLimit) and (hSpeed <= self._speedLimit):

              # Calculate the next state given current state and speed
              nextStateIndex = [i + vSpeed, j + hSpeed, self._encodeSpeed([vSpeed, hSpeed])]

              actionValue = 0

              # Calculate action value given the next state
              if not self._traceCrossesBorder((i, j), (nextStateIndex[0], nextStateIndex[1])):

                # Agent didn't move over the track boundaries
                actionValue = self.stateValues[tuple(nextStateIndex)] - self._transitionCost
              else:
                # Agent hits the boundary
                actionValue = dropOffActionValue

              if actionValue > newStateValues[i, j, k]:
                newStateValues[i, j, k] = actionValue
                if policy is not None: policy[i, j, k] = action

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

  def _traceCrossesBorder(self, startPos, endPos):
    if not endPos[0] in range(self.stateValues.shape[0]) or \
       not endPos[1] in range(self.stateValues.shape[1]):

      return True

    dv, dh = endPos[0] - startPos[0], endPos[1] - startPos[1]

    if dv != 0:
      step = 1 if dv >= 0 else -1
      for v in range(startPos[0] + step, endPos[0] + step, step):
        h = int(round((dh / dv) * (v - startPos[0])) + startPos[1])

        if self.schema[v, h] == 0: return True

    if dh != 0:
      step = 1 if endPos[1] >= startPos[1] else -1
      for h in range(startPos[1] + step, endPos[1] + step, step):
        v = int(round((dv / dh) * (h - startPos[1])) + startPos[0])

        if self.schema[v, h] == 0: return True

    return False

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
      self.policy = np.zeros(self.stateValues.shape, dtype=np.ndarray)
      newStateValues = self.valueIteration(self.policy)
      delta = np.abs((self.stateValues - newStateValues).sum())
      self.stateValues = newStateValues
      t += 1
      print('######    Iteration %2d, Delta: %3.3f    ######' % (t, delta))


  def getNextAction(self, vPos, hPos, vSpeed, hSpeed):
    if not hasattr(self, 'policy'):
      self.findOptimalPolicy()

    return self.policy[vPos, hPos, self._encodeSpeed([vSpeed, hSpeed])]


# Example of usage
policy = PolicyEvaluation()
r1 = policy._borderCrossesMatrix(dv, dh)


# policy.bootstrap()
# print(policy.getNextAction(0, 5, 0, 0)[0])
