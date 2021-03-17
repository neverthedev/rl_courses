from race import Race
from solutioners.policy_evaluation import PolicyEvaluation

filename = 'traces/trace_big'
game = Race(traceFile = filename, visualize = True)
game.run(solutioner = PolicyEvaluation(filename))
