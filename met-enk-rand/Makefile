MONGO=mongodb://hyperopt:hyperopt@kahana.mongohq.com:10090/hyperopt-mixtape
EXP_KEY=$(shell sha1sum experiment.py | awk '{print $$1}')

startserver:
	PYTHONPATH='.' python -c 'from experiment import main; main()' --mongo=$(MONGO) --exp-key=$(EXP_KEY)

runworker:
	PYTHONPATH='.' hyperopt-mongo-worker --mongo=$(MONGO) --exp-key=$(EXP_KEY) --workdir=workdir

interact:
	python -c 'import sys, IPython; from hyperopt.mongoexp import MongoTrials; trials=MongoTrials(sys.argv[1], sys.argv[2]); IPython.embed()' $(MONGO) $(EXP_KEY)

saveresults:
	python -c 'import sys; import json; from bson.json_util import default; from hyperopt.mongoexp import MongoTrials; trials = MongoTrials(sys.argv[1], sys.argv[2]); json.dumps(trials.trials, open("results-%s.json" % sys.argv[2][:7], "w"), default=default)' $(MONGO) $(EXP_KEY)
