from __future__ import print_function
from __future__ import division
import os
import datetime as dt
from tqdm import tqdm as ProgressBar
from collections import defaultdict, Counter
import numpy as np
import sys
sys.path.append('../eda')
import utilities as ut


class NgramModel(object):
    """
    PURPOSE: To train and evaluate a smoothed ngram model

    ARGS:
    comments        (list(str)) list of comments
    """

    def __init__(self,comments):
        self._vocab_counts_(comments)
        self.gram_frequency = None

    def _vocab_counts_(self,comments):
        """
        PURPOSE: Perform initial counting of uni/bi/trigrams present in all comments
                 grouped by classification

        ARGS:
        comments        (tuple) of the following
            comments[0]     (str)       unique comment id
            comments[1]     (list(str)) comments as list of tokens
            comments[2]     (int)       class
        """
        # Attribute Setup
        self.unigram_counts = [Counter(),Counter()]
        self.bigram_counts = [defaultdict(lambda: Counter()),
                              defaultdict(lambda: Counter())]
        self.trigram_counts = [defaultdict(lambda:  Counter()),
                               defaultdict(lambda:  Counter())]
        self.label_counts = [0,0]

        # Token Processing
        for _,comment,label in ProgressBar(comments, desc='Processing Comments'):
            self.label_counts[label] += 1
            prev_unigram = None
            prev_bigram = None
            for word in comment:
                self.unigram_counts[label][word] += 1
                if prev_unigram is not None:
                    self.bigram_counts[label][prev_unigram][word] += 1
                    if prev_bigram is not None:
                        self.trigram_counts[label][prev_bigram][word] += 1
                if prev_unigram is not None:
                    prev_bigram = '_'.join([prev_unigram,word])
                prev_unigram = word

        self.label_probs = [ self.label_counts[i]/sum(self.label_counts) for i in [0,1]]

        # Uni/Bi/Trigram Totals
        self.unigram_totals = [sum(self.unigram_counts[i].values()) for i in [0,1]]
        self.bigram_totals = [sum([sum(endgram_counts.values())
                                    for endgram_counts in self.bigram_counts[i].values()])
                                for i in [0,1]]
        self.trigram_totals = [sum([sum(endgram_counts.values())
                                     for endgram_counts in self.trigram_counts[i].values()])
                                for i in [0,1]]



    def _additive_smoothing_(self,gram_length=1,param=1,counts=False,logs=True):
        """
        PURPOSE: Calculate the smoothed frequencies of the requested uni/bi/trigrams

        ARGS:
        gram_length         (int) gram requested (1) unigram, (2) bigram, (3) trigram
        param               (float) smoothing parameter
        counts              (bool) indicator for  raw counts instead of freq returned
        logs                (bool) indicator for whether log frequencies returned

        RETURNS:
        gram_frequency      (list(dict)) dictionarys of gram counts/frequencies by class
        """

        def smooth_freq(gram_count,gram_total_count,vocab_count,counts,param,logs):
            """
            PURPOSE: Calculate log probability or gram counts

            ARGS:
            gram_count          (int) number of gram appearances
            gram_total_count    (int) total number of appearances of grams of a given length
            vocab_count         (int) total number of unique grams in training set
            logs                (bool) indicator for log frequencies are returned
            counts              (bool) indicator for raw counts instead of freqs returned

            RETURNS:
            count_or_freq       (int_or_float) raw count (int) or log frequency (float)
            """
            if counts:
                count_or_freq = float(gram_count)
            else:
                if logs:
                    count_or_freq = np.log((param + gram_count)/(param*vocab_count + gram_total_count))
                else:
                    count_or_freq = (param + gram_count)/(param*vocab_count + gram_total_count)

            return count_or_freq

        # Unigram Frequencies
        if gram_length == 1:
            vocab_size = [len(self.unigram_counts[i].keys()) for i in [0,1]]
            gram_frequency = [ { gram:smooth_freq(gram_count,self.unigram_totals[i],
                                                       vocab_size[i],counts,param,logs)
                                      for gram, gram_count in self.unigram_counts[i].items()}
                                     for i in [0,1]]
            # Adding smoothed values for words not in training vocab
            for i in [0,1]:
                gram_frequency[i].update({'<unk>': smooth_freq(1,self.unigram_totals[i],
                                                                 vocab_size[i],
                                                                 counts,param,logs)})

        # Bigram Frequencies
        if gram_length == 2: 
            vocab_size = [sum([len(endgram_counts.keys())
                                for endgram_counts in self.bigram_counts[i].values()])
                            for i in [0,1]]
            gram_frequency = [ [ {'_'.join([gram1,gram2]):smooth_freq(gram_count,
                                                                      self.bigram_totals[i],
                                                                      vocab_size[i],
                                                                      counts,param,logs)
                                        for gram2,gram_count in self.bigram_counts[i].get(gram1).items()}
                                      for gram1 in self.bigram_counts[i].keys() ]
                                    for i in [0,1] ]
            # Flattening list of dicts into one dict
            gram_frequency = [ { k:v for d in gram_frequency[i] for k,v in d.items()}
                                     for i in [0,1]]
            # Adding smoothed values for words not in training vocab
            for i in [0,1]:
                gram_frequency[i].update({'<unk>': smooth_freq(1,self.bigram_totals[i],
                                                                vocab_size[i],
                                                                counts,param,logs)})

        # Trigram Frequencies
        if gram_length == 3:
            vocab_size = [sum([len(endgram_counts.keys())
                                for endgram_counts in self.trigram_counts[i].values()])
                            for i in [0,1]]
            gram_frequency = [ [ {'_'.join([gram1,gram2]):smooth_freq(gram_count,
                                                                      self.trigram_totals[i],
                                                                      vocab_size[i],
                                                                      counts,param,logs)
                                        for gram2,gram_count in self.trigram_counts[i].get(gram1).items()}
                                      for gram1 in self.trigram_counts[i].keys() ]
                                    for i in [0,1] ]
            # Flattening a list of dicts into a one dict
            gram_frequency = [ { k:v for d in gram_frequency[i] for k,v in d.items()} 
                                    for i in [0,1]]
            # Adding smoothed values for words not in training vocab
            for i in [0,1]:
                gram_frequency[i].update({'<unk>': smooth_freq(1,self.trigram_totals[i],
                                                                    vocab_size[i],
                                                                    counts,param,logs)})

        return gram_frequency


    def good_turing(self,gram_length=1,param=0.5,logs=True):
        """
        PURPOSE: Calculate good-turing smoothing ngram frequencies

        ARGS:
        gram_length         (int) model gram length
        param               (float)
        logs                (bool) indicator for whether log frequencies are returned

        RETURNS:
        gram_frequency      (list(dict)) gram frequencies by class
        """
        gram_count_by_occur_count = [Counter(),Counter()]
        gram_smoothing_ratios  = [defaultdict(int),defaultdict(int)]
        gram_frequency = [defaultdict(float),defaultdict(float)]
        gram_counts =  self._additive_smoothing_(gram_length,counts=True)

        if logs:
            freq = lambda x: np.log(x)
        else:
            freq = lambda x: x

        for i in [0,1]:
            total = sum(gram_counts[i].values())

            for _,val in gram_counts[i].items():
                gram_count_by_occur_count[i][val] += 1

            for val in gram_count_by_occur_count[i].keys():
                gram_smoothing_ratios[i][val] = (gram_count_by_occur_count[i].get(val+1,
                                                     param*gram_count_by_occur_count[i].get(val))
                                                        /gram_count_by_occur_count[i].get(val))

            for key,val in gram_counts[i].items():
                gram_frequency[i][key] = freq((val+1)*gram_smoothing_ratios[i].get(val)/total)
        gram_frequency[0].default_factory = None
        gram_frequency[1].default_factory = None

        return gram_frequency


    def _gram_freq_(self,gram,gram_models):
        """
        PURPOSE: Look up the frequecy of a uni/bi/trigram in the gram_models

        ARGS:
        gram        (str) gram whose frequency is generated
        gram_models (list(dict)) uni/bi/trigram models source

        RETURNS:
        probs       (list(float)) gram_probability by label
        """
        num_tokens = len(gram.split('_'))
        probs = [ gram_models[num_tokens-1][i].get(gram,0) for i in [0,1]]
        return probs


    def _smoothed_gram_freqs_(self,gram,gram_models,cnvx_param=0.5,logs=True):
        """
        PURPOSE: Implementing the smoothed convex combination of higher and lower gram
                 probabilities, consistent with the jelinek-mercer smoothing

        ARGS:
        gram            (str) an "_" seperated uni/bi/trigram
        gram_models     (list(dict)) frequency distributions for all uni/bi/trigrams
        cnvxs_param     (float) convex combination parameter
        logs            (bool) indicator for if probability is given in logs or levels

        RETURNS:
        gram_freq       (list(float)) list of log or level gram frequencies by class
        """
        num_tokens = len(gram.split('_'))
        class_gram_freq = [1/sum(self.unigram_counts[i].values()) for i in [0,1]]
        unigrams = gram.split('_')

        convex_comb = lambda param,arg1,arg2: param*arg1 + (1-param)*arg2

        if num_tokens >=1:
            unigram_freq = [ [ convex_comb(cnvx_param,
                                           self._gram_freq_(gram,gram_models)[i],
                                           class_gram_freq[i])
                                for gram in unigrams]
                              for i in [0,1]]
            gram_freq = [val for sublist in unigram_freq for val in sublist]

        if num_tokens >=2:
            bigrams = ['_'.join(unigrams[i:i+2]) for i in range(num_tokens-1)]

            unigram_freq_products = [ [ np.prod(unigram_freq[j][i:i+2])
                                        for i in range(num_tokens-1)]
                                      for j in [0,1]]
            bigram_freq = [ [ convex_comb(cnvx_param,
                                           self._gram_freq_(bigrams[j],gram_models)[i],
                                           unigram_freq_products[i][j])
                                for j,_ in enumerate(bigrams)]
                              for i in [0,1]]
            gram_freq = [val for sublist in bigram_freq for val in sublist]

        if num_tokens ==3:
            trigram = gram

            bigram_freq_products = [ np.prod(bigram_freq[j]) for j in [0,1]]

            trigram_freq = [ convex_comb(cnvx_param,
                                         self._gram_freq_(trigram,gram_models)[i],
                                         bigram_freq_products[i])
                             for i in [0,1]]
            gram_freq = trigram_freq

        if logs:
            return [ np.log(val) for val in gram_freq]
        else:
            return gram_freq


    def jelinek_mercer(self,gram_length=1,cnvx_param=0.5,scnd_smoother='additive',**kwargs):
        """
        PURPOSE: Generate a gram frequency models using the convex combinations
                 of uni/bi/trigram frequency models generated in accordance with
                 the secondary smoother.

        ARGS:
        gram_length         (int) gram length of final model
        cnvx_param          (float) parameter used to generate convex combinations
        scnd_smoother       (str) secondary smoothing technique to generate gram models

        KWARGS: Parameters needed for secondary smoothers

        RETURNS:
        gram_frequency      (list(dict)) smoothed gram model by label
        """
        gram_models = []
        unigram_vocab_counts =[len(self.unigram_counts[i]) for i in [0,1]]

        if scnd_smoother ==  'additive':
            param = kwargs.get('param',1)
            for gram_model in range(1,gram_length+1):
                gram_models.append(self._additive_smoothing_(gram_model,param,
                                                             counts=False,logs=False))

        elif scnd_smoother == 'good_turing':
            param = kwargs.get('param',0.5)
            for gram_model in range(1,gram_length+1):
                gram_models.append(self.good_turing(gram_model,param,logs=False))

        vocab = set(gram_models[-1][0].keys()).union(gram_models[-1][1].keys())

        gram_frequency = [{},{}]
        for gram in ProgressBar(vocab, desc='Processing Gram'):
            prob0,prob1 = self._smoothed_gram_freqs_(gram,gram_models,cnvx_param)
            gram_frequency[0].update({gram:prob0})
            gram_frequency[1].update({gram:prob1})

        return gram_frequency


    def train_classifier(self,gram_length=1,smoothing='additive',**kwargs):
        """
        PURPOSE: Train a Naive Bayes Classifier with frequencies generated by requested
                 smoothing technique.

        ARGS:
        gram_length         (int) model gram length
        smoothing           (str) requested smoothing technique

        KWARGS: parameters necessary for requested smoothing function
        scnd_smoother       (str) name of secondary smoother used in jelinek_mercer
        smth_param          (float) parameter passed to the primary or secondary smoother
        cnvx_param          (float) convex combination parameter used in jelinek_mercer
        """
        self.gram_length = gram_length
        self.smoothing_info = {'smoothing_method':smoothing}
        self.smoothing_info.update(kwargs)

        if smoothing == 'additive':
            param = kwargs.get('smth_param',1)
            self.gram_frequency = self._additive_smoothing_(gram_length,param)
        elif smoothing == 'good-turing':
            param = kwargs.get('smth_param',0.5)
            self.gram_frequency = self.good_turing(gram_length,param)
        elif smoothing == 'jelinek-mercer':
            cnvx_param = kwargs.get('cnvx_param',0.5)
            smth_param = kwargs.get('smth_param',1)
            scnd_smoother = kwargs.get('scnd_smoother','additive')
            self.gram_frequency = self.jelinek_mercer(gram_length,cnvx_param,
                                                      scnd_smoother,param=smth_param)
        else:
            print('Smoothing technique {} not recognized'.format(smoothing))


    def predict(self,comment):
        """
        PURPOSE: Predict the classification of a tokenized comment with a ngram
                 model of the specified length with the specified smoothing technique

        ARGS:
        comment             (list(str)) list of tokens representing the comment
        gram_length         (int) gram freq requested (1) unigram, (2) bigram, (3) trigram
        smoothing           (str) smoothing technique used in ['additive']
        **kwargs            parameters needed for the indicated smoothing method

        RETURNS:
        predicted           (int) predicted class
        log_probabilities   (list(float)) calculated log likelihood
        """
        if self.gram_frequency is None:
            print('Error Model Not Trained')

        unk_prob = [0,0]
        if self.gram_length == 1:
            log_probs = [ sum([ self.gram_frequency[i].get(gram,unk_prob[i])
                                for gram in comment])
                          for i in [0,1]]
        elif self.gram_length == 2:
            comment_grams = ['_'.join(comment[i:i+2]) for i in range(len(comment)-1)]
            log_probs = [ sum([ self.gram_frequency[i].get(gram,unk_prob[i])
                                for gram in comment_grams])
                          for i in [0,1]]
        elif self.gram_length == 3:
            comment_grams = ['_'.join(comment[i:i+3]) for i in range(len(comment)-2)]
            log_probs = [ sum([ self.gram_frequency[i].get(gram,unk_prob[i])
                                for gram in comment_grams])
                         for i in [0,1]]

        predicted = int(log_probs[0] + np.log(self.label_probs[0]) < \
                        log_probs[1] + np.log(self.label_probs[1]))

        return predicted, log_probs


    def evaluate_classifier(self,comments,labels,report=True,file=False):
        """
        PURPOSE: Evaluate the currently trained model

        ARGS:
        comments        (list(list)) list of tokenized comments
        labels          (list)
        """
        from sklearn.metrics import classification_report, confusion_matrix
        import json

        predicted_labels = []

        for comment in comments:
            predicted_label, predicted_log_prob = self.predict(comment)
            predicted_labels.append(predicted_label)

        if report:
            print('------{}------\n{}'.format("Confusion Matrix",
                                               confusion_matrix(labels,predicted_labels)))
            print('------{}------\n{}'.format("Report",
                                               classification_report(labels,
                                                                     predicted_labels)))

        if file:
            now_time = str(dt.datetime.now().microsecond)[:4]
            summary_dir = ''.join([os.getcwd(),'/ngram_summary'])
            file_ext = ''.join(['NG-',self.smoothing_info['smoothing_method'],
                                  '-',str(self.gram_length),
                                  '-',now_time])
            self.summary_file = ''.join([summary_dir,'/',file_ext,'.json'])

            summary_dict = {}
            summary_dict.update(self.smoothing_info)
            class_report_dict = classification_report(labels,predicted_labels,
                                                      output_dict=True)
            summary_dict.update(class_report_dict)
            with open(self.summary_file,'w') as file:
                json.dump(summary_dict,file,indent=2)

