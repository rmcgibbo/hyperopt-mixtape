from hyperopt import tpe, fmin, Trials
from hyperopt.mongoexp import MongoTrials
from objective import fit_and_score
from modelspace import space

trials = MongoTrials('mongo://localhost:1234/met-enk/jobs', exp_key='exp1')
# trials = Trials()
best = fmin(fit_and_score, space, trials=trials, algo=tpe.suggest, max_evals=100)
