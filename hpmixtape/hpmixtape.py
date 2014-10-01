import numpy as np
from hyperopt import base, pyll
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.kernel_approximation import Nystroem as _Nystroem

def trial_to_fninval(trial, space):
    """Instantiate a particular element from the pyll space

    Example
    -------
    >>> space = ...
    >>> trials = ...
    >>> fmin(fn, space=space, trials=trials)

    # Now, let's say we want to get the input value
    # to `fn` that was used in the first trial:
    >>> inval = trial_to_fninval(trails.trials[0], space)

    # and then, e.g.:
    >>> fn(inval)
    """
    # this code comes from hyperopt.base.Domain.evaluate and
    # hyperopt.base.Domain.memo_from_config
    config = base.spec_from_misc(trial['misc'])
    memo = {}
    for node in pyll.dfs(pyll.as_apply(space)):
        if node.name == 'hyperopt_param':
            label = node.arg['label'].obj
            memo[node] = config.get(label, pyll.base.GarbageCollected)
    return pyll.rec_eval(space, memo=memo)

def modelFactory(arg):
    assert modelFactory == arg.pop('_factory')
    klass = arg.pop('_class')
    model = klass(**arg)
    return model 

#def atomPairsFactory(arg):
#    assert atomPairsFactory == arg.pop('_factory')
#    assert AtomPairsFeaturizer == arg.pop('_class')
#    n_pairs = arg.pop('_n_pairs')
#    n_atoms = arg.pop('_n_atoms')
#    seed = arg.pop('_pairs_seed')
#    random = np.random.RandomState(seed)
#    pairs = set()
#    while len(pairs) < n_pairs:
#        fs = frozenset(random.randint(n_atoms, size=2))
#        if len(fs) == 2:
#            pairs.add(fs)
#    pairs = [tuple(e) for e in pairs]
#    return AtomPairsFeaturizer(pairs, **arg)


def pipelineFactory(args):
    steps = []
    for s in args['steps']:
        if isinstance(s, dict) and '_factory' in s:
            step = s['_factory'](s)
            steps.append((step.__class__.__name__, step))
        elif isinstance(s, BaseEstimator):
            steps.append((s.__class__.__name__, s))
    return Pipeline(steps)


class Nystroem(_Nystroem):
    def fit(self, sequences, y=None):
        return super(Nystroem, self).fit(np.concatenate(sequences))

    def transform(self, sequences):
        trans = super(Nystroem, self).transform
        y = [trans(X) for X in sequences]
        return y

