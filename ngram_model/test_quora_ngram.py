import unittest
import numpy as np
from collections import defaultdict
from quora_ngram import NgramModel 

class TestQuoraNgramCounts(unittest.TestCase): 

    def setUp(self):
        comments = [['xxxxx',['where','are','you'],1],['yyyyy',['who','are','you'],0]]
        self.vocab = NgramModel(comments)

    def tearDown(self):
        self.vocab = None

    def test_unigram_counts(self): 
        unigram_counts_target = [{'where': 1, 'are': 1, 'you':1},
                                 {'who': 1, 'are': 1, 'you':1}]
        self.assertDictEqual(self.vocab.unigram_counts[0],unigram_counts_target[1])
        self.assertDictEqual(self.vocab.unigram_counts[1],unigram_counts_target[0])
        self.assertEqual(self.vocab.unigram_totals,[3,3])

    def test_bigram_counts(self): 
        bigram_counts_target = [{'where':{'are':1},
                                'are':{'you':1}},
                                {'who':{'are':1},
                                'are':{'you':1}}]
        self.assertDictEqual(dict(self.vocab.bigram_counts[0]),bigram_counts_target[1])
        self.assertDictEqual(dict(self.vocab.bigram_counts[1]),bigram_counts_target[0])
        self.assertEqual(self.vocab.bigram_totals,[2,2])


    def test_trigram_counts(self): 
        trigram_counts_target = [{'where_are':{'you':1}},
                                 {'who_are':{'you':1}}]
        self.assertDictEqual(dict(self.vocab.trigram_counts[0]),trigram_counts_target[1])
        self.assertDictEqual(dict(self.vocab.trigram_counts[1]),trigram_counts_target[0])
        self.assertEqual(self.vocab.trigram_totals,[1,1])


class TestQuoraNgramAdditiveFreq(unittest.TestCase): 

    def setUp(self):
        comments = [['xxxxx',['where','are','you'],1],['yyyyy',['who','are','you'],0]]
        self.vocab = NgramModel(comments)

    def tearDown(self):
        self.vocab = None

    def test_unigram_freq(self): 
        self.vocab.train_classifier(gram_length=1,smoothing='additive',param=1)
        unigram_freqs_target = [{'where':np.log((1+1)/(1*3+3)),
                                  'are': np.log((1+1)/(1*3+3)),
                                  'you':np.log((1+1)/(1*3+3)),
                                  '<unk>':np.log((1+1)/(1*3+3))},
                                 {'who':np.log((1+1)/(1*3+3)),
                                  'are': np.log((1+1)/(1*3+3)),
                                  'you':np.log((1+1)/(1*3+3)),
                                  '<unk>':np.log((1+1)/(1*3+3))}]
        self.assertDictEqual(self.vocab.gram_frequency[0],unigram_freqs_target[1])
        self.assertDictEqual(self.vocab.gram_frequency[1],unigram_freqs_target[0])

    def test_bigram_freq(self): 
        self.vocab.train_classifier(gram_length=2,smoothing='additive',param=1)
        bigram_freqs_target = [{'where_are':np.log((1+1)/(1*2+2)),
                                  'are_you': np.log((1+1)/(1*2+2)),
                                  '<unk>':np.log((1+1)/(1*2+2))},
                                 {'who_are':np.log((1+1)/(1*2+2)),
                                  'are_you': np.log((1+1)/(1*2+2)),
                                  '<unk>':np.log((1+1)/(1*2+2))}]
        self.assertDictEqual(self.vocab.gram_frequency[0],bigram_freqs_target[1])
        self.assertDictEqual(self.vocab.gram_frequency[1],bigram_freqs_target[0])


    def test_trigram_freq(self):
        self.vocab.train_classifier(gram_length=3,smoothing='additive',param=1)
        trigram_freqs_target = [{'where_are_you':np.log((1+1)/(1*1+1)),
                                  '<unk>':np.log((1+1)/(1*1+1))},
                                 {'who_are_you':np.log((1+1)/(1*1+1)),
                                  '<unk>':np.log((1+1)/(1*1+1))}]
        self.assertDictEqual(self.vocab.gram_frequency[0],trigram_freqs_target[1])
        self.assertDictEqual(self.vocab.gram_frequency[1],trigram_freqs_target[0])

class TestQuoraNgramGTFreq(unittest.TestCase):

    def setUp(self):
        comments = [['xxxxx',['where','where','where','are','are','you','you','now'],1],
                    ['yyyyy',['who','who','who','are','are','you','you','you'],0]]
        self.vocab = NgramModel(comments)

    def tearDown(self):
        self.vocab = None

    def test_unigram_freq(self):
        self.vocab.good_turing(gram_length=1,param=0.5)
        unigram_freqs_target = [{'where':np.log((3+1)*(0.5*1)/(1)/9),
                                 'are':np.log((2+1)*(1)/(2)/9),
                                 'you':np.log((2+1)*(1)/(2)/9),
                                 'now':np.log((1+1)*(2)/(2)/9),
                                 '<unk>':np.log((1+1)*(2)/(2)/9)},
                                {'who':np.log((3+1)*(0.5*2)/(2)/9),
                                 'are':np.log((2+1)*(2)/(1)/9),
                                 'you':np.log((3+1)*(0.5*2)/(2)/9),
                                 '<unk>':np.log((1+1)*(1)/(1)/9)}]
        self.vocab.train_classifier(gram_length=1,smoothing='good-turing')
        self.assertDictEqual(self.vocab.gram_frequency[0],unigram_freqs_target[1])
        self.assertDictEqual(self.vocab.gram_frequency[1],unigram_freqs_target[0])

    def test_bigram_freq(self):
        self.vocab.train_classifier(gram_length=2,smoothing='good-turing')
        bigram_freqs_target = [{'where_where':np.log((2+1)*(0.5*2)/(2)/8),
                                'where_are':np.log((1+1)*(1)/(6)/8),
                                'are_are':np.log((1+1)*(1)/(6)/8),
                                'are_you':np.log((1+1)*(1)/(6)/8),
                                'you_you':np.log((1+1)*(1)/(6)/8),
                               'you_now':np.log((1+1)*(1)/(6)/8),
                                '<unk>':np.log((1+1)*(1)/(6)/8)},
                               {'who_who':np.log((2+1)*(0.5*2)/(2)/8),
                                'who_are':np.log((1+1)*(2)/(4)/8),
                                'are_are':np.log((1+1)*(2)/(4)/8),
                                'are_you':np.log((1+1)*(2)/(4)/8),
                                'you_you':np.log((2+1)*(0.5*2)/(2)/8),
                                '<unk>':np.log((1+1)*(2)/(4)/8)}]
        self.assertDictEqual(self.vocab.gram_frequency[0],bigram_freqs_target[1])
        self.assertDictEqual(self.vocab.gram_frequency[1],bigram_freqs_target[0])

class TestQuoraNgramJMFreq(unittest.TestCase): 
    def setUp(self):
        comments = [['xxxxx',['where','where','where','are','are','you','you','now'],1],
                    ['yyyyy',['who','who','who','are','are','you','you','you'],0]]
        self.vocab = NgramModel(comments)

    def tearDown(self):
        self.vocab = None

    def test_unigram_smoothed_freq(self):
        gram_models = [self.vocab._additive_smoothing_(gram_length=1,param=0,counts=False,logs=False)]
        who_smoothed =[0.5*(3/8+1/8),0.5*(0+1/8)]
        are_smoothed =[0.5*(2/8+1/8),0.5*(2/8+1/8)]
        self.assertListEqual(self.vocab._smoothed_gram_freqs_('who',gram_models,0.5,logs=False),
                             who_smoothed)
        self.assertListEqual(self.vocab._smoothed_gram_freqs_('are',gram_models,0.5,logs=False),
                             are_smoothed)

    def test_bigram_smoothed_freq(self):
        gram_models = [self.vocab._additive_smoothing_(gram_length=1,param=0,counts=False,logs=False),
                       self.vocab._additive_smoothing_(gram_length=2,param=0,counts=False,logs=False)]
        who_smoothed =[0.5*(3/8+1/8),0.5*(0+1/8)]
        are_smoothed =[0.5*(2/8+1/8),0.5*(2/8+1/8)]
        who_are_smoothed = [0.5*(1/7 + who_smoothed[0]*are_smoothed[0]),
                            0.5*(0 + who_smoothed[1]*are_smoothed[1])]
        are_are_smoothed = [0.5*(1/7 + are_smoothed[0]*are_smoothed[0]),
                            0.5*(1/7 + are_smoothed[1]*are_smoothed[1])]

        self.assertListEqual(self.vocab._smoothed_gram_freqs_('who_are',gram_models,0.5,logs=False),
                             who_are_smoothed)
        self.assertListEqual(self.vocab._smoothed_gram_freqs_('are_are',gram_models,0.5,logs=False),
                             are_are_smoothed)

    def test_trigram_smoothed_freq(self):
        gram_models = [self.vocab._additive_smoothing_(gram_length=1,param=0,counts=False,logs=False),
                       self.vocab._additive_smoothing_(gram_length=2,param=0,counts=False,logs=False),
                       self.vocab._additive_smoothing_(gram_length=3,param=0,counts=False,logs=False)]
        self.who_smoothed =[[0.5*(3/8+1/8)],[0.5*(0+1/8)]]
        self.are_smoothed =[[0.5*(2/8+1/8)],[0.5*(2/8+1/8)]]
        self.who_are_smoothed = [[0.5*(1/7 + self.who_smoothed[0][0]*self.are_smoothed[0][0])],[
                            0.5*(0 + self.who_smoothed[1][0]*self.are_smoothed[1][0])]]
        self.are_are_smoothed = [[0.5*(1/7 + self.are_smoothed[0][0]*self.are_smoothed[0][0])],[
                            0.5*(1/7 + self.are_smoothed[1][0]*self.are_smoothed[1][0])]]
        self.who_are_are_smoothed = [[0.5*(1/6 + self.who_are_smoothed[0][0]*self.are_are_smoothed[0][0])],
                                     [0.5*(0 + self.who_are_smoothed[1][0]*self.are_are_smoothed[1][0])]]

        self.assertListEqual(self.vocab._smoothed_gram_freqs_('who_are_are',gram_models,0.5,logs=False),
                             self.who_are_are_smoothed)

if __name__ == '__main__':
    unittest.main()

