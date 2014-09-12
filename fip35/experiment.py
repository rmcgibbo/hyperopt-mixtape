import os
import sys
import time
import argparse
import numpy as np
import mdtraj as md

from hyperopt import hp, tpe, fmin, STATUS_OK
from hyperopt.pyll import scope
from hyperopt.mongoexp import MongoTrials
from hpmixtape import modelFactory, pipelineFactory

from mixtape.pca import PCA
from mixtape.tica import tICA
from mixtape.featurizer import DihedralFeaturizer
from mixtape.markovstatemodel import MarkovStateModel
from mixtape.cluster import MiniBatchKMeans, KCenters, GMM
from mixtape.datasets import fetch_fs_peptide

from sklearn.cross_validation import KFold
from sklearn.externals.joblib import Memory


TRAJECTORIES = None
def load_trajectories():
    pdb = md.load('/home/rmcgibbo/datasets/Fip35-WW/structures/ww_native.pdb')
    heavy = pdb.top.select_atom_indices('heavy')
    pdb = pdb.atom_slice(heavy)
    trajectories = [md.load('/home/rmcgibbo/datasets/Fip35-WW/trj0.lh5', stride=50, atom_indices=heavy),
                    md.load('/home/rmcgibbo/datasets/Fip35-WW/trj1.lh5', stride=50, atom_indices=heavy)]

    # split each trajectory into 3 chunks
    out = []
    for t in trajectories:
        t.top = pdb.top
        n = len(t) / 3
        for i in range(0, len(t), n):
            chunk = t[i:i+n]
            if len(chunk) > 1:
                out.append(chunk)

    print([len(t) for t in out])
            
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mongo', required=True)
    parser.add_argument('--exp-key', required=True)
    args = parser.parse_args()

    # from hyperopt import Trials
    # trials = Trials()

    trials = MongoTrials(args.mongo, exp_key=args.exp_key)
    best = fmin(fit_and_score, modelspace, trials=trials, algo=tpe.suggest, max_evals=500)


modelspace = {'_factory': pipelineFactory,
         'steps': [
    hp.choice('featurization', [
        {'_class': DihedralFeaturizer,
         '_factory': modelFactory,
         'types' : ['phi', 'psi'],
         'sincos': True},
        {'_class': DihedralFeaturizer,
         '_factory': modelFactory,
         'types': ['phi', 'psi', 'chi1'],
         'sincos': True},
        {'_class': DihedralFeaturizer,
         '_factory': modelFactory,
         'types': ['phi', 'psi', 'chi1', 'chi2'],
         'sincos': True},
    ]),
    hp.choice('preprocessing', [
        {'_class': PCA,
         '_factory': modelFactory,
         'n_components': scope.int(hp.quniform('pca_n_components', 2, 20, 1)),
         'copy': False},
        {'_class': tICA,
         '_factory': modelFactory,
         'n_components': scope.int(hp.quniform('tica_n_components', 2, 20, 1)),
         'gamma': hp.choice('tica_gamma', [0, 1e-7, 1e-5, 1e-3, 1e-1]),
         'weighted_transform': hp.choice('tica_weighted_transform', [True, False])
       }
    ]),
    hp.choice('cluster', [
        {'_class': MiniBatchKMeans,
         '_factory': modelFactory,
         'n_clusters': scope.int(hp.quniform('kmeans_n_clusters', 10, 1000, 10)),
         'batch_size': 10000,
         'n_init': 1,
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
    global TRAJECTORIES
    if TRAJECTORIES is None:
        TRAJECTORIES = load_trajectories()
    model = model_spec['_factory'](model_spec)

    parameters = {k:v for k, v in model.get_params().items() if '__' in k}
    train_scores, test_scores, fit_times = [], [], []
    cv = KFold(len(TRAJECTORIES), n_folds=3)
    for fold, (train_index, test_index) in enumerate(cv):
        train_data = [TRAJECTORIES[i] for i in train_index]
        test_data = [TRAJECTORIES[i] for i in test_index]

        start = time.time()
        try:
            model.fit(train_data)
        except:
            print(model)
            raise
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
