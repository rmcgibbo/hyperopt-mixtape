startdb:
	mongod --dbpath db --port 1234 --directoryperdb --journal --nohttpinterface --logpath mongo-logfile.txt &

startserver:
	python optimize.py &

runworker:
	PYTHONPATH='.' hyperopt-mongo-worker --mongo=localhost:1234/met-enk --poll-interval=0.1 --workdir=workdir

