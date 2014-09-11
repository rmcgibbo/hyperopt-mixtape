MONGO=mongodb://hyperopt:hyperopt@kahana.mongohq.com:10090/hyperopt-mixtape
EXP_KEY=$(shell sha1sum experiment.py | awk '{print $$1}')

startserver:
	PYTHONPATH='.' python -c 'from experiment import main; main()' --mongo=$(MONGO) --exp-key=$(EXP_KEY)

runworker:
	PYTHONPATH='.' hyperopt-mongo-worker --mongo=$(MONGO) --exp-key=$(EXP_KEY) --workdir=workdir

