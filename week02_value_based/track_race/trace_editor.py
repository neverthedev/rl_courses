import sys, os
sys.path.append(os.getcwd())

from race import Race

game = Race('traces/trace_big')
game.runEditorMode()
