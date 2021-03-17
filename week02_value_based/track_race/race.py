import pygame, sys
import numpy as np
from os import path
from colors import Colors
from gamefield import GameField
from pdb import set_trace as debugger

class Race:
  def __init__(self, traceFile=None, visualize=False):
    self.visualize = visualize

    if traceFile:
      self.traceFileName = traceFile
    else:
      self.traceFileName = 'traces/trace_big.dat'

    if self.loadTrace():
      self.setRandomInitialState()

    self._reward = 0

  def loadTrace(self):
    if path.exists(self.traceFileName):
      self.schema = np.loadtxt(self.traceFileName, dtype=np.int)
      self._height, self._width = self.schema.shape
      return True
    else:
      self._height, self._width = 50, 150
      self.schema = np.zeros((self._height, self._width), dtype=np.int)
      return False

  def setRandomInitialState(self):
    self._initial_position_y = self._position_y = self.height - 1
    available_positions = np.where(self.schema[self._position_y]  == 1)[0]
    self._initial_position_x = self._position_x = np.random.choice(available_positions, 1)[0]
    self._speed_x = self._speed_y = 0

  @property
  def height(self):
    return self._height

  @property
  def width(self):
    return self._width

  @property
  def gamefield(self):
    if not hasattr(self, '_gamefield'):
      self._gamefield = GameField(self.height, self.width, self.editorMode)
      self._gamefield.init()
    return self._gamefield

  @property
  def is_final_step(self):
    return self.schema[self._position_y, self._position_x] == 2

  @property
  def reward(self):
    return self._reward

  @property
  def state(self):
    return (self._position_x, self._position_y, self._speed_x, self._speed_y)

  def makeAction(self):
    if self.is_final_step:
      if self.visualize: self.gamefield.quit()
      return 0

    delta_y, delta_x  = self.solutioner.getNextAction(
      self.height - self._position_y - 1, self._position_x, self._speed_y, self._speed_x
    )

    self._speed_x = min(5, max(-5, self._speed_x + delta_x))
    self._speed_y = min(5, max(-5, self._speed_y + delta_y))

    new_pos_x = self._position_x + self._speed_x
    new_pos_y = self._position_y - self._speed_y # start point is at the bottom left corner

    if (new_pos_x < 0) or (new_pos_x >= self.width) or (new_pos_y < 0) or (new_pos_y >= self.height) or (self.schema[new_pos_y, new_pos_x] == 0):
      self.setRandomInitialState()
      self._reward += -100
    else:
      self._position_x = new_pos_x
      self._position_y = new_pos_y
      self._reward += -1

    if self.visualize: self.displayCurrentGameState()

    return -1

  def displayCurrentGameState(self):
    self.gamefield.setScheme(self.schema, redraw=False)
    if not self.editorMode:
      self.gamefield.setPlayerPosition(self._position_x, self._position_y, redraw=False)
      self.gamefield.setPlayerSpeed(self._speed_x, self._speed_y, redraw=False)
      self.gamefield.setInformation(self._reward, redraw=False)
    self.gamefield.redraw()

  def runEditorMode(self):
    self.editorMode = True
    self.gamefield.subscribeEvent('exit', self.saveAndExit)
    self.gamefield.subscribeEvent('selection', self.updateSelectedArea)
    self.displayCurrentGameState()
    self.gamefield.run()

  def run(self, solutioner):
    self.editorMode = False
    self.solutioner = solutioner
    if self.solutioner: self.solutioner.bootstrap()

    if self.visualize:
      self.displayCurrentGameState()
      self.gamefield.subscribeEvent('exit', self.printFinalScoreAndExit)
      self.gamefield.subscribeEvent('space', self.makeAction)
      self.gamefield.run()
    else:
      while not self.is_final_step:
        self.makeAction()

    print('Final reward is %d' % self._reward)

  def updateSelectedArea(self, r_rows, r_cols, delta):
    self.schema[r_rows[0]:(r_rows[1] + 1), r_cols[0]:(r_cols[1] + 1)] += delta
    self.schema = np.mod(self.schema, 3)
    self.gamefield.setScheme(self.schema)

  def saveAndExit(self):
    np.savetxt(self.traceFileName, self.schema, '%d')
    self.gamefield.quit();
    sys.exit();

  def printFinalScoreAndExit(self):
    print('Final reward is %d' % self._reward)
    self.gamefield.quit();
