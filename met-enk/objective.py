import time
import numpy as np
from hyperopt import STATUS_OK
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold

from mixtape.datasets import fetch_met_enkephalin

def pipelineFactory(args):
    steps = []
    for arg in args:
        if arg is not None:
            if isinstance(arg, dict) and 'class' in arg:
                klass = arg.pop('class')
                step = klass(**arg)
                steps.append((step.__class__.__name__, step))
    return Pipeline(steps)


def fit_and_score(model_spec):
    model = pipelineFactory(model_spec)
    dataset = fetch_met_enkephalin()
    trajectories = dataset['trajectories']

    parameters = {k:v for k, v in model.get_params().items() if '__' in k}
    train_scores, test_scores, fit_times = [], [], []
    cv = KFold(len(trajectories), n_folds=3)
    for fold, (train_index, test_index) in enumerate(cv):
        train_data = [trajectories[i] for i in train_index]
        test_data = [trajectories[i] for i in test_index]

        start = time.time()
        model.fit(train_data)
        fit_times.append(time.time()-start)
        train_scores.append(model.score(train_data))
        test_scores.append(model.score(test_data))

    result = {'loss': -np.mean(test_scores),
              'status': STATUS_OK,
              'train_scores': train_scores,
              'test_scores': test_scores,
              'parameters': parameters,
              'fit_times': fit_times}
    print(result)
    return result
