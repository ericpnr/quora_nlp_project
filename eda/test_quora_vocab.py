import unittest
from collections import defaultdict
from quora_vocab import CommentVocab

class TestQuoraVocab(unittest.TestCase): 

    def setUp(self):
        comments = [['xxxxx',['where','are','you'],1],['yyyyy',['who','are','you'],0]]
        self.vocab = CommentVocab(comments)

    def tearDown(self):
        self.vocab = None

    def test_unigram_counts(self): 
        unigram_counts_target = [{'<s>':1,'where': 1, 'are': 1, 'you':1,'</s>':1},
                                 {'<s>':1,'who': 1, 'are': 1, 'you':1,'</s>':1}]
        self.assertDictEqual(self.vocab.unigram_counts[0],unigram_counts_target[1])
        self.assertDictEqual(self.vocab.unigram_counts[1],unigram_counts_target[0])

    def test_bigram_counts(self): 
        bigram_counts_target = [{'<s>':{'where':1},
                                'where':{'are':1},
                                'are':{'you':1},
                                'you':{'</s>':1}},
                                {'<s>':{'who':1},
                                'who':{'are':1},
                                'are':{'you':1},
                                'you':{'</s>':1}}]
        self.assertDictEqual(dict(self.vocab.bigram_counts[0]),bigram_counts_target[1])
        self.assertDictEqual(dict(self.vocab.bigram_counts[1]),bigram_counts_target[0])


if __name__ == '__main__': 
    unittest.main()

