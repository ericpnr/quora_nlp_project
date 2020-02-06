from __future__ import print_function
from __future__ import division
from tqdm import tqdm as ProgressBar
import numpy as np
import utilities as ut
from collections import defaultdict, Counter

class CommentVocab(object):
    """
    PURPOSE: Generate a vocabulary, uni/bi/tri gram counts, and plotting tools for 
             a quora comments

    ARGS:
        comments        (tuple) of the following
            comments[0]     (str)       unique comment id
            comments[1]     (list(str)) comments as list of tokens
            comments[2]     (int)       class
    """

    def __init__(self,comments,size=None):
        self.UNK_TOKEN = '<unk>'
        self.VOCAB_SIZE = size
        self._vocab_counts_(comments)
        self._comment_lengths_(comments)

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
        #Attribute Setup
        self.unigram_counts = [Counter(),Counter()]
        self.bigram_counts = [defaultdict(lambda: Counter()),
                              defaultdict(lambda: Counter())]
        self.trigram_counts = [defaultdict(lambda:  Counter()),
                               defaultdict(lambda:  Counter())]
        self.label_counts = [0,0]

        #Token Processing
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

        #Uni/Bi/Trigram Totals
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
        gram_length         (int) gram frequencies requested (1) unigram, (2) bigram, (3) trigram
        param               (float) smoothing parameter
        counts              (bool) indicator for whether raw counts instead of freq are returned
        logs                (bool) indicator for whether log frequencies are returned

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
            logs                (bool) indicator for whether log frequencies are returned
            counts              (bool) indicator for whether raw counts instead of freqs are returned

            RETURNS:
            count_or_freq       (int_or_float) either a raw count (int) or a log frequency (float)
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
                                                                      vocab_size[i],counts,param,logs)
                                        for gram2,gram_count in self.bigram_counts[i].get(gram1).items()}
                                      for gram1 in self.bigram_counts[i].keys() ]
                                    for i in [0,1] ]
            # Flattening list of dicts into one dict
            gram_frequency = [ { k:v for d in gram_frequency[i] for k,v in d.items()}
                                     for i in [0,1]]
            # Adding smoothed values for words not in training vocab
            for i in [0,1]:
                gram_frequency[i].update({'<unk>': smooth_freq(1,self.bigram_totals[i],
                                                                vocab_size[i],counts,param,logs)})

        # Trigram Frequencies
        if gram_length == 3:
            vocab_size = [sum([len(endgram_counts.keys())
                                for endgram_counts in self.trigram_counts[i].values()])
                            for i in [0,1]]
            gram_frequency = [ [ {'_'.join([gram1,gram2]):smooth_freq(gram_count,
                                                                      self.trigram_totals[i],
                                                                      vocab_size[i],counts,param,logs)
                                        for gram2,gram_count in self.trigram_counts[i].get(gram1).items()}
                                      for gram1 in self.trigram_counts[i].keys() ]
                                    for i in [0,1] ]
            # Flattening a list of dicts into a one dict
            gram_frequency = [ { k:v for d in gram_frequency[i] for k,v in d.items()}
                                    for i in [0,1]]
            # Adding smoothed values for words not in training vocab
            for i in [0,1]:
                gram_frequency[i].update({'<unk>': smooth_freq(1,self.trigram_totals[i],
                                                                    vocab_size[i],counts,param,logs)})

        return gram_frequency


    def _comment_lengths_(self,comments):
        """
        PURPOSE: Calculate the lenght of the comments given

        ARGS:
        comments        (tuple) of the following
            comments[0]     (str)       unique comment id
            comments[1]     (list(str)) comments as list of tokens
            comments[2]     (int)       class
        """
        comment_lengths = [Counter([len(comment[1]) for comment in comments if comment[2] == i])
                                for i in range(2)]
        comment_lengths = [zip(comment_lengths[0].keys(),comment_lengths[0].values()),
                           zip(comment_lengths[1].keys(),comment_lengths[1].values())]
        self.sorted_comment_lengths = [sorted(comment_lengths[0],key = lambda x: x[0]),
                                       sorted(comment_lengths[1],key = lambda x: x[0])]


    def word_frequency_graphs(self,gram_length=1,min_rank=0,max_rank=25):
        """
        PURPOSE: Generate a word gram frequency in accordance with the requested
                 gram  length

        ARGS:
        gram_length     (int) gram model requested
        min_rank        (int) lowest rank (by gram frequency) shown (low pass filter)
        max_rank        (int) highest rank (by gram frequency) show (high pass filter)

        RETURNS:        (figure) bokeh figure containing requested graph
        """
        from bokeh.plotting import figure
        from bokeh.io import show
        from bokeh.models import HoverTool, ColumnDataSource

        if gram_length==1:
            gram_freq = self._additive_smoothing_(gram_length,param = 0, counts=True,logs=False)
            title = 'Unigram'
        elif gram_length==2:
            gram_freq = self._additive_smoothing_(gram_length,param = 0, counts=True,logs=False)
            title = 'Bigram'
        elif gram_length==3:
            gram_freq = self._additive_smoothing_(gram_length,param = 0, counts=True,logs=False)
            title = 'Bigram'

        neg_vocab_words,neg_vocab_freq= list(zip(*sorted(zip(gram_freq[0].keys(),gram_freq[0].values()),
                                               key=lambda x: -x[1])[min_rank:max_rank]))
        pos_vocab_words,pos_vocab_freq= list(zip(*sorted(zip(gram_freq[1].keys(),gram_freq[1].values()),
                                               key=lambda x: -x[1])[min_rank:max_rank]))

        source = ColumnDataSource(data=dict(neg_vocab_words=neg_vocab_words,
                                            neg_vocab_freq=neg_vocab_freq,
                                            pos_vocab_words=pos_vocab_words,
                                            pos_vocab_freq=pos_vocab_freq))

        hover_1 = HoverTool(tooltips=[('Word', "$neg_vocab_words"),('Freq','$neg_vocab_freq')])
        hover_2 = HoverTool(tooltips=[('Word', "$pos_vocab_words"),('Freq','$pos_vocab_freq')])

        p1 = figure(x_range=neg_vocab_words,plot_height=300,plot_width=750,
                    tools=[hover_1],title="Appropriate Question {} Counts".format(title))
        p2 = figure(x_range=pos_vocab_words,plot_height=300,plot_width=750,
                    tools=[hover_2],title="Inappropriate Question {} Counts".format(title))

        p1.vbar(x='neg_vocab_words',top='neg_vocab_freq',width=0.9,source=source)
        p2.vbar(x='pos_vocab_words',top='pos_vocab_freq',width=0.9,source=source)

        p1.xgrid.grid_line_color = None
        p2.xgrid.grid_line_color = None

        p1.xaxis.major_label_orientation = 1
        p2.xaxis.major_label_orientation = 1

        show(p1)
        show(p2)


    def _gram_differences_(self,gram_length=1):
        """
        PURPOSE: Calculate the differences in gram_frequencies between classes

        ARGS:
            gram_length     (int) 1=unigram,2=bigram,3=trigram

        RETURNS:
            gram_freq_diffs (list(tup)) sorted list of (gram, frequency difference)
        """
        gram_freq = self._additive_smoothing_(gram_length, param=0,counts=False,logs=False)
        vocab = set(gram_freq[0].keys()).union(gram_freq[1].keys())

        gram_freq_diffs = []
        for gram in vocab:
            gram_freq_diff = gram_freq[0].get(gram,0) - gram_freq[1].get(gram,0)
            gram_freq_diffs.append((gram,gram_freq_diff))

        gram_frequency_diffs = sorted(gram_freq_diffs,key=lambda x: -x[1])
        return gram_frequency_diffs


    def word_count_difference_graph(self,gram_length=1,num=10):
        """
        PURPOSE: Generate a plot of the most extreme differences in uni/bigram usage by label

        ARGS:
        gram_length         (int) unigram(1) bigram(2) or trigram(3) difference shown
        num                 (int) number of comparisons from each tail shown

        RETURNS:
        p                   (plot) bokeh figure with requested plot
        """
        from bokeh.plotting import figure
        from bokeh.io import show

        gram_diff = self._gram_differences_(gram_length)
        words,diff = zip(*(gram_diff[:num] + gram_diff[-num:]))

        if gram_length ==1:
            title = 'Unigrams'
        elif gram_length ==2:
            title = 'Bigrams'
        elif gram_length ==3:
            title = 'Trigrams'

        p = figure(x_range = words,
                   plot_height=300,
                   plot_width=750,
                   title='{} {}: ({} - {})'.format(title,'Frequency Difference',
                                                         'Appropriate Frequency',
                                                         'Inappropriate Frequency'))

        p.vbar(x=words,top=diff,width=0.9)
        p.xgrid.grid_line_color = None
        p.xaxis.major_label_orientation = 1

        show(p)


    def comment_length_graph(self):
        """
        PURPOSE: Generate a graph with two superimpose histograms comparing the frequency of
                 comment lengths across labels

        RETURNS:
        p           (plot) plot bokeh figure containing graph
        """
        from bokeh.plotting import figure
        from bokeh.io import show
        from bokeh.models import HoverTool, ColumnDataSource

        neg_len, neg_len_counts = zip(*self.sorted_comment_lengths[0])
        pos_len, pos_len_counts = zip(*self.sorted_comment_lengths[1])
        max_length = np.max((neg_len[-1],pos_len[-1]))

        neg_len_freq_plotting = [ neg_len_counts[neg_len.index(i)]/sum(neg_len_counts)
                                    if i in neg_len else 0
                                    for i in range(max_length+1)]
        pos_len_freq_plotting = [ pos_len_counts[pos_len.index(i)]/sum(pos_len_counts)
                                    if i in pos_len else 0
                                    for i in range(max_length+1)]
        data = {'lengths' : list(range(max_length+1)),
                'neg': neg_len_freq_plotting,
                'pos': pos_len_freq_plotting}

        colors = ["#c9d9d3", "#718dbf"]

        p = figure(plot_width=750, plot_height=400,
                   title='Distribution of Question Lengths')
        p.vbar(x='lengths',top='pos',
               width=0.9,source=data,
               color=colors[1],alpha=0.75,
               legend='Inappropriate')
        p.vbar(x='lengths',top='neg',
               width=0.9,source=data,
               color=colors[0],alpha=0.75,
               legend='Appropriate')

        show(p)

