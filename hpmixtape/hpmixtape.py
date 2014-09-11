from sklearn.pipeline import Pipeline


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
    return Pipeline(steps)