.PHONY: all clean load preprocess # train inference 

PYTHON = python3

all: load #results/solution/solution.csv


# Load data
data/raw :load

	mkdir data/raw -p
	# Train
	wget \
		[link to data] \
		-p data/raw/
	unzip \
		-q data/raw/train_data.zip\
		-d data/raw
	rm data/raw/train_data.zip

# Delete everything except raw data and code
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -r data/processed
	rm -r data/interim
	rm -r logs/
	rm -r models/
	rm -r results/