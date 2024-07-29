compile:
	python3 scripts/lib/utils/compile.py;
sa-train:
	python3 scripts/bin/sa-model.py;
train:
	python3 scripts/bin/train-large.py;
test:
	python3 scripts/bin/test-model.py;
make plot:
	python3 scripts/bin/plot.py;
cook-fineweb:
	python3 scripts/bin/cooking/fineweb-edu-score-2.py;
compress:
	python3 scripts/bin/compress.py;