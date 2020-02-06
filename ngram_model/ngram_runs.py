import os
import sys
from importlib import reload

import numpy as np
import pandas as pd
from random import shuffle

sys.path.append('../embeddings')
sys.path.append('../eda')
import utilities as ut
import quora_ngram as ngram

# NOTE: The purpose of the following script is to train and evaluate a specific setup 
#       of the quora_ngram.py 

# Data Loading and Tokenization
DATA_PATH = '../eda/'
DATA_FILE = '{}{}'.format(DATA_PATH,'train.csv')
data = pd.read_csv(DATA_FILE)

data_pos_tokenized = [[vec[0],ut.canon_token_sentence(vec[1]),vec[2]]
                      for vec in data.to_numpy() if vec[2] == 1]
data_neg_tokenized = [[vec[0],ut.canon_token_sentence(vec[1]),vec[2]]
                      for vec in data.to_numpy() if vec[2] == 0]

# Variables
NVALID = 5000
NTRAIN = 150000

#Data Extraction and Splitting
train_data = (data_pos_tokenized[NVALID:NTRAIN+NVALID] +
              data_neg_tokenized[NVALID:NTRAIN+NVALID])
shuffle(train_data)
valid_data = data_pos_tokenized[:NVALID] + data_neg_tokenized[:NVALID]
shuffle(valid_data)

train_comments = [tpl[1] for tpl in train_data]
valid_comments = [tpl[1] for tpl in valid_data]
labels_train = [val[2] for val in train_data]
labels_valid = [val[2] for val in valid_data]

ng_classifier = ngram.NgramModel(train_data)
smth_parameters = [ i/10 for i in range(1,11)]

for gram in [1,2,3]:
    for param in smth_parameters:
        ng_classifier.train_classifier(gram_length=gram,
                                       smoothing='jelinek-mercer',
                                       smth_param=param,
                                       cnvx_param=0.5,
                                       scnd_smoother='additive')
        ng_classifier.evaluate_classifier(valid_comments,
                                          labels_valid,
                                          report=True,file=True)
