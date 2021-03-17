from race import Race
from solutioners.policy_evaluation import PolicyEvaluation

game = Race(visualize=True)
game.run(solutioner = PolicyEvaluation())
