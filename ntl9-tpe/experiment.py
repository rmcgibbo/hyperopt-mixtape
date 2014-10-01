import os
import sys
import time
import glob
import argparse
import numpy as np

from hyperopt import hp, tpe, rand, fmin, STATUS_OK, Trials
from hyperopt.pyll import scope
from hyperopt.mongoexp import MongoTrials
from hpmixtape import modelFactory, pipelineFactory

from mixtape.tica import tICA
from mixtape.markovstatemodel import MarkovStateModel
from mixtape.cluster import KMeans, KCenters

from sklearn.cross_validation import KFold
from sklearn.externals.joblib import Memory

FEAT_DIR = '/home/rmcgibbo/datasets/Voelz_2009_abInitioProteinFolding/features-stride10/'
FEAT_GLOB = 'traj-*-dihedral-ppc1.pkl'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mongo', required=True)
    parser.add_argument('--exp-key', required=True)
    args = parser.parse_args()

    trials = MongoTrials(args.mongo, exp_key=args.exp_key)
    best = fmin(fit_and_score, modelspace, trials=trials, algo=rand.suggest, max_evals=500)


modelspace = {'_factory': pipelineFactory,
         'steps': [
    hp.choice('preprocessing', [
        None,
        {'_class': tICA,
         '_factory': modelFactory,
         'n_components': scope.int(hp.quniform('tica_n_components', 2, 20, 1)),
         'gamma': hp.choice('tica_gamma', [0, 1e-7, 1e-5, 1e-3, 1e-1]),
         'weighted_transform': hp.choice('tica_weighted_transform', [True, False])
       }
    ]),
    hp.choice('cluster', [
        {'_class': KMeans,
         '_factory': modelFactory,
         'n_clusters': scope.int(hp.quniform('kmeans_n_clusters', 10, 500, 10)),
         'n_init': 1,
        },
        {'_class': KCenters,
         '_factory': modelFactory,
         'n_clusters': scope.int(hp.quniform('kcenters_n_clusters', 10, 500, 10)),
        },
    ]),
    {'_class': MarkovStateModel,
     '_factory': modelFactory,
     'verbose': False,
     'n_timescales': 3,
     'reversible_type': 'transpose'
    },
]}


def fit_and_score(model_spec):
    model = model_spec['_factory'](model_spec)

    def load_features():
        import joblib
        features = [joblib.load(fn) for fn in glob.iglob(os.path.join(FEAT_DIR, FEAT_GLOB))]
        return features
    trajectories = load_features()

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
