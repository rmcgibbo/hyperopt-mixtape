from hyperopt import hp
from hyperopt.pyll import scope

from mixtape.tica import tICA
from mixtape.featurizer import DihedralFeaturizer
from mixtape.markovstatemodel import MarkovStateModel
from mixtape.cluster import KMeans, KCenters

space = [
    hp.choice('featurization', [
        {'class': DihedralFeaturizer,
         'types' : ['phi', 'psi'],
         'sincos': True},
        {'class': DihedralFeaturizer,
         'types': ['phi', 'psi', 'chi1'],
         'sincos': True}
    ]),
    hp.choice('preprocessing', [
        None,
        {'class': tICA,
         'n_components': scope.int(hp.quniform('tica_n_components', 2, 20, 1)),
         'gamma': hp.choice('tica_gamma', [0, 1e-7, 1e-5, 1e-3, 1e-1]),
         'weighted_transform': hp.choice('tica_weighted_transform', [True, False])
       }
    ]),
    hp.choice('cluster', [
        {'class': KMeans,
         'n_clusters': scope.int(hp.quniform('kmeans_n_clusters', 10, 500, 10)),
         'n_init': 1,
        },
        {'class': KCenters,
         'n_clusters': scope.int(hp.quniform('kcenters_n_clusters', 10, 500, 10)),
        },
    ]),
    {'class': MarkovStateModel,
     'verbose': False,
     'n_timescales': 3,
     'reversible_type': 'transpose'
    },
]
