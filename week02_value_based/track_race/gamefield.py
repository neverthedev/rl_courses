import pygame, math, sys, random
from colors import Colors
import pdb

class GameField:

  def __init__(self, rowsCnt, colsCnt, editorMode):
    self.callbacks = {}

    self.editorMode = editorMode

    self.borderWidth = 2
    self.infoPaneWidth = 600

    self.left, self.top, self.scale = 0, 0, 10

    self.rowsCnt = rowsCnt
    self.colsCnt = colsCnt

    self.width = self.colsCnt * self.scale
    self.height = self.rowsCnt * self.scale

    self.selectionStart = None
    self._is_run = False

  @property
  def widthOnScreen(self):
    width = self.width + 2 * self.borderWidth
    if not self.editorMode:
      width += self.infoPaneWidth + self.borderWidth

    return width

  @property
  def heightOnScreen(self):
    return self.height + 2 * self.borderWidth

  @property
  def clock(self):
    if not hasattr(self, '_clock'):
      self._clock = pygame.time.Clock()
    return self._clock

  def setPlayerPosition(self, x, y, redraw=True):
    self.playerPosition_x, self.playerPosition_y = x, y
    if self._is_run and redraw:
      self.redraw()

  def setScheme(self, scheme, redraw=True):
    self.scheme = scheme
    if self._is_run and redraw:
      self.redraw()

  def setPlayerSpeed(self, speedX, speedY, redraw=True):
    self._speedX = speedX
    self._speedY = speedY
    if self._is_run and redraw:
      self.redraw()

  def setInformation(self, reward, redraw=True):
    self._reward = reward
    if self._is_run and redraw:
      self.redraw()

  def getSchemeRange(self, pos_start, pos_end):
    start_x = math.floor((pos_start[0] - self.left - self.borderWidth) / self.scale)
    end_x = math.floor((pos_end[0] - self.left - self.borderWidth) / self.scale)
    start_y = math.floor((pos_start[1] - self.top - self.borderWidth) / self.scale)
    end_y = math.floor((pos_end[1] - self.top - self.borderWidth) / self.scale)

    range_x = (start_x, end_x) if start_x <= end_x else (end_x, start_x)
    range_y = (start_y, end_y) if start_y <= end_y else (end_y, start_y)

    return (range_x, range_y)

  def init(self):
    pygame.init()
    pygame.display.set_caption('Trace race')
    self.screen = pygame.display.set_mode((self.widthOnScreen, self.heightOnScreen))

  def quit(self):
    self._is_run = False
    pygame.quit()

  def subscribeEvent(self, event, callback):
    self.callbacks[event] = callback

  def run(self):
    while self._is_run:
      self.processEvents()

      self.clock.tick(10)

      if 'clock-tick' in self.callbacks:
        self.callbacks['clock-tick']()

  def processEvents(self):
    for event in pygame.event.get():

      if event.type == pygame.QUIT:
        if 'exit' in self.callbacks:
          self.callbacks['exit']()

      if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
          if 'exit' in self.callbacks:
            self.callbacks['exit']()
        if event.key == pygame.K_SPACE:
          if 'space' in self.callbacks:
            self.callbacks['space']()

      if event.type == pygame.MOUSEBUTTONDOWN:
        self.selectionStart = event.pos

      if event.type == pygame.MOUSEBUTTONUP:
        if self.selectionStart:
          r_cols, r_rows = self.getSchemeRange(self.selectionStart, event.pos)
          if event.button == 1:
            delta = 1
          elif event.button == 3:
            delta = -1
          else:
            delta = 0

          if 'selection' in self.callbacks:
            self.callbacks['selection'](r_rows, r_cols, delta)

  def redraw(self):
    self._is_run = True

    self.screen.fill(Colors.WHITE)

    self.drawTraceBorder()
    self.drawScheme()
    if not self.editorMode:
      self.drawInfoPaneBorder()
      self.drawPlayer()
      self.drawInfoPane()
    pygame.display.flip()

  def drawScheme(self):
    for rowIdx, row in enumerate(self.scheme):
      for colIdx, item in enumerate(row):

        pos1 = ((self.left + (self.scale * colIdx) + self.borderWidth, self.top + self.scale * (rowIdx) + self.borderWidth),
               (self.scale, self.scale))

        pos2 = ((self.left + (self.scale * colIdx) + 1 + self.borderWidth, self.top + self.scale * (rowIdx) + 1 + self.borderWidth),
               (self.scale - 1, self.scale - 1))

        rect = pygame.draw.rect(self.screen, Colors.BLACK, pos1, 0)

        if item == 0:
          color = Colors.RED
        elif item == 1:
          color = Colors.GREEN
        elif item == 2:
          color = Colors.BLUE

        rect = pygame.draw.rect(self.screen, color, pos2, 0)

  def drawTraceBorder(self):
    points = [(self.left, self.top),
              (self.left + self.width + self.borderWidth, self.top),
              (self.left + self.width + self.borderWidth, self.top + self.height + self.borderWidth),
              (self.left, self.top + self.height + self.borderWidth)]

    pygame.draw.lines(self.screen, Colors.BLACK, True, points, self.borderWidth)

  def drawInfoPaneBorder(self):
    points = [(self.left + self.width + self.borderWidth, self.top),
              (self.left + self.width + 2 * self.borderWidth + self.infoPaneWidth, self.top),
              (self.left + self.width + 2 * self.borderWidth + self.infoPaneWidth, self.top + self.height + self.borderWidth),
              (self.left + self.width + self.borderWidth, self.top + self.height + self.borderWidth)]

    pygame.draw.lines(self.screen, Colors.BLACK, True, points, self.borderWidth)

  def drawPlayer(self):
    pos = ((self.left + (self.scale * self.playerPosition_x) + 2 + self.borderWidth,
             self.top + (self.scale * self.playerPosition_y) + 2 + self.borderWidth),
            (self.scale - 4, self.scale - 4))

    rect = pygame.draw.rect(self.screen, Colors.BLACK, pos, 0)


  def drawInfoPane(self):
    startLeft = self.left + self.width + self.borderWidth
    startTop = self.top + 20

    headerFont = pygame.font.SysFont('freeserif', 40, bold=True)
    rowHeaderFont = pygame.font.SysFont('freeserif', 25, bold=True)
    rowRegularFont = pygame.font.SysFont('freeserif', 25)

    caption = headerFont.render('Game information', True, (0, 0, 0))
    self.screen.blit(caption, (startLeft + 60, startTop))
    startTop += 60

    xSpeedCaption = rowHeaderFont.render('Horizontal speed:', True, (0, 0, 0))
    self.screen.blit(xSpeedCaption, (startLeft + 20, startTop))
    xSpeedValue = rowRegularFont.render("%d" % self._speedX, True, (0, 0, 0))
    self.screen.blit(xSpeedValue, (startLeft + 250, startTop))

    startTop += 40

    ySpeedCaption = rowHeaderFont.render('Vertical speed:', True, (0, 0, 0))
    self.screen.blit(ySpeedCaption, (startLeft + 20, startTop))
    ySpeedValue = rowRegularFont.render("%d" % self._speedY, True, (0, 0, 0))
    self.screen.blit(ySpeedValue, (startLeft + 250, startTop))

    startTop += 40

    rewardCaption = rowHeaderFont.render('Current reward:', True, (0, 0, 0))
    self.screen.blit(rewardCaption, (startLeft + 20, startTop))
    rewardValue = rowRegularFont.render("%d" % self._reward, True, (0, 0, 0))
    self.screen.blit(rewardValue, (startLeft + 250, startTop))
