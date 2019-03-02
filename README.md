# Automatic Gapping Resolution for Russian (AGRR-2019)
This repository contains a model for automatic gapping resolution for Russian and it was evaluated during Dialogue Evaluation 2019 (https://github.com/dialogue-evaluation/AGRR-2019).
Basically, the repository contains two independent models: the first model is a binary classifier (presence/absence of gapping) based on BiGRU and the second model is a multilabel classifier based on the Universal Transformer (encoder) that solves a sequence labeling task by assigning a label to each word in a sentence with gapping. 

## Prerequirements
* Download fastText model for Russian at https://fasttext.cc/docs/en/crawl-vectors.html and specify the path to the binary file in ```settings.py```.
* Install russian punkt model for nltk at https://github.com/Mottl/ru_punkt .

## Usage
Use ```train.py``` for training and ```eval.py``` for evaluating. The input for the evaluation script should be a csv file in the same format as test data released by the organizers. The results will be written in ```results/parsed.csv```.
