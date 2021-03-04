import numpy as np
import sys, os
from pdb import set_trace as debugger

sys.path.append(os.getcwd())

from race import Race
import random as rand

class PolicyEvaluation:

  def __init__(self):
    self._speedLimit = 5
    self._speedsCount = self._speedLimit * 2 + 1
    self._dropOffPenalty = 100
    self._transitionCost = 1

    # Load track schema
    self.traceFileName = os.getcwd() + '/traces/trace_cut.dat'
    self.schema = np.flip(np.loadtxt(self.traceFileName, dtype=np.int), axis=0)

    # Initialize state values function with arbitrary values
    self.stateValues = np.zeros(
      (self.schema.shape[0], self.schema.shape[1],  self._speedsCount ** 2)
    )
    # Generate full set of actions
    self._actionsMatrix = self._getActionsMatrix()

  def _getActionsMatrix(self):
    incs = [-1, 0, 1]
    return np.array([[ [i, j] for j in incs ] for i in incs])

  def valueIteration(self):
    newStateValues = np.zeros(self.stateValues.shape)

    for i in range(self.stateValues.shape[0]):
      for j in range(self.stateValues.shape[1]):
        for k in range(self.stateValues.shape[2]):
          newStateValues[i, j, k] = - np.inf
          speedsMatrix = self._actionsMatrix + self._decodeSpeed(k)
          for vSpeed, hSpeed in speedsMatrix.reshape(-1, 2):
            finalSpeed = self._encodeSpeed([vSpeed, hSpeed])

            if (vSpeed != 0 or hSpeed != 0) and \
               (vSpeed >= - self._speedLimit) and (vSpeed <= self._speedLimit) and \
               (hSpeed >= - self._speedLimit) and (hSpeed <= self._speedLimit):

              nextStateIndex = [i + vSpeed, j + hSpeed, finalSpeed]

              actionValue = 0

              if nextStateIndex[0] in range(self.stateValues.shape[0]) and \
                 nextStateIndex[1] in range(self.stateValues.shape[1]) and \
                 nextStateIndex[2] in range(self.stateValues.shape[2]) and \
                 self.schema[nextStateIndex[0], nextStateIndex[1]] != 0:

                debugger()
                # Agent didn't move over the boundary
                if self.schema[nextStateIndex[0], nextStateIndex[1]] == 2:
                  actionValue = 10
                elif self.schema[nextStateIndex[0], nextStateIndex[1]] == 1:
                  actionValue = self.stateValues[tuple(nextStateIndex)] - self._transitionCost
                end
              else:

                initialStates = self.initialStates()

                for hPos in initialStates:
                  actionValue += self.stateValues[0, hPos, self._encodeSpeed([0, 0])]
                actionValue = actionValue / len(initialStates) - self._dropOffPenalty

              newStateValues[i,j,k] = max(actionValue,  newStateValues[i,j,k])
          #debugger()
          #123

  def _encodeSpeed(self, speedVector):
    return (speedVector[0] + self._speedLimit) * self._speedsCount  + \
           (speedVector[1] + self._speedLimit)

  def _decodeSpeed(self, speed):
    vSpeed = speed // self._speedsCount - self._speedLimit
    hSpeed = speed % self._speedsCount - self._speedLimit

    return np.array([vSpeed, hSpeed])

  def initialStates(self):
    return np.array(np.where(self.schema[0] == 1)[0])


  def valueIterations(self):
    delta, t  = 100, 0
    print('######    Iteration 0   ######')
    while delta > 1.:
      newStateValues = self.valueIteration()
      delta = np.abs((self.stateValues - newStateValues).sum())
      self.stateValues = newStateValues
      t += 1
      print('######    Iteration %2d, Delta: %3.3f    ######' % (t, delta))


policy = PolicyEvaluation()
policy.valueIterations()
debugger()
123
