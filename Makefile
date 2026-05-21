
.PHONY: install data lint test train report clean

install:
	pip install -r requirements.txt

data:
	python src/data/download.py

lint:
	flake8 src/ tests/ --max-line-length=100

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

train:
	python src/models/train.py

report:
	python src/visualization/report.py

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f data/processed/*.parquet
