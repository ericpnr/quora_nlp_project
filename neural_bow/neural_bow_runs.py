import os
import sys
from importlib import reload
import numpy as np
import pandas as pd
from random import shuffle
sys.path.append('../embeddings')
sys.path.append('../eda')
import embedding_utilities as embd_util
import utilities as ut
import neural_bow_classifier as nbow

# NOTE: The purpose of the following script is to train and evaluate a specific setup 
#       of the neural_bow_classifier.py 

# Data Loading and Tokenization
DATA_PATH = '~/google_drive/data/quora/'
DATA_FILE = '{}{}'.format(DATA_PATH,'train.csv')
data_full = pd.read_csv(DATA_FILE)
data = data_full
data_pos_tokenized = [[vec[0],ut.canon_token_sentence(vec[1]),vec[2]]
                      for vec in data.to_numpy() if vec[2] == 1]
data_neg_tokenized = [[vec[0],ut.canon_token_sentence(vec[1]),vec[2]]
                      for vec in data.to_numpy() if vec[2] == 0]

# Variables
NVALID = 5000
NTRAIN = 100000
MAX_SEQ_LEN = 100
EMBD_DIM = 100

# Data Extraction and Splitting
train_data = (data_pos_tokenized[NVALID:NTRAIN+NVALID] +
              data_neg_tokenized[NVALID:NTRAIN+NVALID])
shuffle(train_data)
valid_data = data_pos_tokenized[:NVALID] + data_neg_tokenized[:NVALID]
shuffle(valid_data)

train_comments = [tpl[1] for tpl in train_data]
valid_comments = [tpl[1] for tpl in valid_data]
labels_train = [val[2] for val in train_data]
labels_valid = [val[2] for val in valid_data]

# Word Embeddings Extraction
filename = '../embeddings/quora_full_embeds.txt'
embeddings, token_to_row_number = embd_util.embedding_extraction(filename,
                                                                 embedding_length=EMBD_DIM)
embeddings.append([0]*EMBD_DIM)
embeddings=np.array(embeddings)

# Conversion of Training and Validation Data to Embedding id sequences
train_id_sequences = embd_util.token_sequence_to_id_sequence(train_comments,
                                                             token_to_row_number,
                                                             unknown_token='UNK',
                                                             max_sequence_length=100)
valid_id_sequences = embd_util.token_sequence_to_id_sequence(valid_comments,
                                                             token_to_row_number,
                                                             unknown_token='UNK',
                                                             max_sequence_length=100)

# Class instantiation and graph building
nbow_classifier = nbow.Neural_BOW({'n_tokens':len(embeddings)-1,
                                   'max_sequence_length':100,
                                   'embedding_dimension':EMBD_DIM})
nbow_classifier.build_graph()

# Training
train_dict = {'embeddings':embeddings,
              'sequences_train':train_id_sequences,'labels_train': labels_train,
              'sequences_valid':valid_id_sequences,'labels_valid': labels_valid,'n_stop':1}
nbow_classifier.train_graph(train_dict)

# Reporting
nbow_classifier.predict_and_report(valid_id_sequences,labels_valid,embeddings,file=True)
