hyperopt-mixtape
================

Experiments/WIP for using [hyperopt](https://github.com/hyperopt/hyperopt)
and [mixtape](https://github.com/rmcgibbo/mixtape) together to optimize Markov
state models.

Install
-------
1. Get my [hacked version](https://github.com/rmcgibbo/hyperopt) of hyperopt.
2. Get the latest version of mixtape
3. Start a mongodb server. I'm currently using [compose](https://www.compose.io/),
   which is a hosted mongodb that's free. This seemed easier, because there
   are more complicated IP routing issues if you try to run the mongo server on
   localhost, and you don't want to deal with that.
4. Go into the `hpmixtape` directory in this project, and install it.


Running
-------
the job of the server is basically to orchestrate the experiment and choose
the parameter settings for the workers. boot up the server with:
```
$ make startserver
```

then run one or more workers
```
$ make runworker
$ make runworker
# etc...
$ make runworker
```

Some of this is automated by the `run-experiment.sh` script, which submits
a job which make


Files
-----

Each experiment in this project is one of the subdirectories, and it includes
three key files

- `experiment.py`
    This file contains all the science. The search space is declared inside as
    a prior distribution over MSM hyperparameters, defined using hyperopt
    syntax and some unfortunate factory classes. The objective function (e.g.
    loading the trajectory data and running the cross validation) is declared
    inside. Finally, there's the main function, which sets up the optimization
    and calls `fmin`.
    
    Note: When `main()` gets called, the system just blocks. "Workers" need to
    connect to the minimizer to actually get anything _done_.

- `Makefile`
    The makefile mostly serves to shorten otherwise long and unwieldy command
    line invocations.
    
    Inside the makefile, we set the "experiment key", which is a unique identifier
    for each "experiment". The experiment is identified by the sha1 hash of 
    the 'experiment.py' file. Each MSM that is built during the course of the
    experiment is tagged in the database with this experiment key, and workers
    are targeted towards the appropriate project by specifying the connection
    string to the mongo DB instance and a particular experiment key.
    
- `run-experiment.sh`
    Simple PBS script to automate the process of starting the server and then
    running many workers on a cluster.