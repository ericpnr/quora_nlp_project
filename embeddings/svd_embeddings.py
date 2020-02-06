from __future__ import print_function
from __future__ import division

from collections import defaultdict, Counter
import numpy as np
import scipy.sparse
from tqdm import tqdm as ProgressBar
from sklearn.decomposition import TruncatedSVD

class SVD_Embeddings(object):
    """
    PURPOSE: Generate embedding vectors for the unique set of tokens (vocabulary) contained
             within a training set consisting of a list of ordered sequences of tokens.
             Embeddings are generated via Singular Value Decomposition.

    ARGS:
    sequences               (list(str)) or (list(list(str)) strings or tokenized list
    embedding_dimension     (int) length of token embedding vectors
    max_sequence_length     (int) longest token sequence allowed
    tokenized               (bool) indicator for whehter sequences are tokenized lists
    """
    def __init__(self,sequences,embedding_dimension=100,max_sequence_length=100,tokenized=False):
        """
        PURPOSE: Initial setup and token occurance/cooccurance counting

        """
        #Attribute Setup
        self.EMBD_DIM = embedding_dimension
        self.UNK_NAME = 'UNK'
        self.tokenized=tokenized
        self.transformer = None
        self.MAX_LEN = max_sequence_length
        self.trained_embeddings = None
        self.token_counts = Counter()
        self.co_token_counts = defaultdict(lambda: Counter())
        for sequence in ProgressBar(sequences,desc='Token Counts'):
            if tokenized:
                sequence_list = sequence
            else:
                sequence_list = sequence.split()
            for token1 in sequence_list:
                self.token_counts[token1] += 1
                for token2 in set(sequence_list).difference({token1}):
                    self.co_token_counts[token1][token2] += 1

        self.id_to_token = dict(enumerate(self.token_counts.keys()))
        self.UNK_ID = np.max(list(self.id_to_token.keys()))+1
        self.token_to_id = {v:k for k,v in self.id_to_token.items()}

        self.vocab_size = len(self.id_to_token)
        self.total_count = sum(self.token_counts.values())

        self.train_sequences_w_pad = self.sequences_to_ids_w_pad(sequences,
                                                                 self.MAX_LEN,
                                                                 tokenized=self.tokenized)


    def sequences_to_ids(self,sequences,tokenized=False):
        """
        PURPOSE: Convert a string of white space seperated token names into id numbers

        ARGS:
        sequences       (list(list(str)) ordered sequences of tokens for conversion of id numbers
        tokenized       (bool) indicator for whether the sequence is a list of tokens

        RETURNS:
        sequences_with_ids      (list(list(int))) ordered sequences of ids numbers
        """
        sequences_with_ids = []
        for sequence in ProgressBar(sequences):
            if tokenized:
                sequence_list = sequence
            else:
                sequence_list = sequence.split()

            for item in sequence_list:
                item_id = self.token_to_id.get(item,self.UNK_ID)
                sequence_with_ids += [item_id]
            sequences_with_ids += sequence_with_ids

        return sequences_with_ids


    def sequences_to_ids_w_pad(self,sequences,max_sequence_len=50,tokenized=False):
        """
        PURPOSE: Converting sequences to id's then padding to self.MAX_LEN

        ARGS:
        sequences           (list(list)) sequences of tokens for conversion
        max_sequence_len    (int) maximum length of sequence to bee padded to

        RETURNS:
        sequences_w_pad     (list(list(int))) sequences of ids padded to the max tokens num
        """
        sequences_w_pad = []
        for sequence in ProgressBar(sequences,desc="Conversion to ID"):
            if tokenized:
                sequence_list = sequence
            else:
                sequence_list = sequence.split()

            pre_sequence_ids = self.tokens_to_ids(sequence_list)
            sequence_ids_w_pad = pre_sequence_ids[:self.MAX_LEN]\
                                 + [self.UNK_ID]*(self.MAX_LEN - len(pre_sequence_ids))
            sequences_w_pad.append(sequence_ids_w_pad)

        return sequences_w_pad


    def tokens_to_ids(self, tokens):
        """
        PURPOSE: Convert a list of string token names to ids

        ARGS:
        tokens      (list(str)) list of tokens for conversion to ids

        RETURNS:
        ids         (list(int)) list of integer id numbers for tokens
        """
        ids = [self.token_to_id.get(token,self.UNK_ID) for token in tokens]
        return ids


    def ids_to_tokens(self, ids):
        """
        PURPOSE: Generate token names from a list of token ids

        ARGS:
        ids         (list(int)) list of ids for conversion to tokens

        RETURNS:
        tokens      (list(str)) list of tokens
        """
        tokens = [self.id_to_token[id] for id in ids]
        return tokens


    def pointwise_mutual_info(self,token,token_list):
        """
        PURPOSE: Generate a vector of the pointwise mutual information values between
                 the token and all tokens in the token_list

        ARG:
        token         (str) target token for pmi calculation
        token_list    (list(str)) list of tokens

        RETURNS:
        pmi            (list(float)) vector of pmi values for each token in token list
        """
        pmi = []
        target_token_count = self.token_counts.get(token)
        for co_token in token_list:
            token_co_token_count = self.co_token_counts.get(token).get(co_token)
            co_token_count = self.token_counts.get(co_token)
            pre_pmi= np.log((token_co_token_count*self.total_count)/
                            (target_token_count*co_token_count))
            pmi.append(np.max([0,pre_pmi]))

        return pmi


    def co_occurance_matrix(self,pmi=True):
        """
        PURPOSE: Generating a sparse co-purchase matrix

        ARGS:
        pmi        (bool) indicator for whether pmi will be calculated
        """
        rows = []
        columns = []
        data = []
        for token in ProgressBar(self.co_token_counts.keys(),desc='Co Occurance Matrix'):
            token_id = self.token_to_id.get(token)
            co_tokens_ids = self.tokens_to_ids(self.co_token_counts.get(token).keys())
            co_tokens_counts = self.co_token_counts.get(token).values()
            rows += [token_id]*len(co_tokens_ids)
            columns += co_tokens_ids
            if pmi:
                data += self.pointwise_mutual_info(token,self.co_token_counts.get(token).keys())
            else:
                data += co_tokens_counts

        self.co_occurance = scipy.sparse.csc_matrix((data,(rows,columns)),
                                                   shape=(self.vocab_size,self.vocab_size))


    def SVD_train(self):
        """
        PURPOSE: Training a Truncated SVD for the Class
        """
        self.transformer = TruncatedSVD(n_components=self.EMBD_DIM, random_state=1)
        self.co_occurance_matrix()
        raw_trained_embeddings = self.transformer.fit_transform(self.co_occurance)
        #Normalize to unit length
        normed_trained_embeddings = raw_trained_embeddings / np.linalg.norm(raw_trained_embeddings,
                                                                            axis=1).reshape([-1,1])
        self.trained_embeddings = np.vstack((normed_trained_embeddings,
                                             np.array([0]*self.EMBD_DIM)))
