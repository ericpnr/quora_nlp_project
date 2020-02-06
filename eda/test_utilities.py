import unittest
import utilities as ut

class TestUtilities(unittest.TestCase):

    def test_canonicalize_sentence(self):
        self.assertEquals(ut.canonicalize_sentence('word.other'),'word other')
        self.assertEquals(ut.canonicalize_sentence('word. other'),'word other')
        self.assertEquals(ut.canonicalize_sentence('word/ other'),'word other')
        self.assertEquals(ut.canonicalize_sentence('word/other'),'word other')
        self.assertEquals(ut.canonicalize_sentence('word other?'),'word other')

    def test_canon_sentence(self):
        input = 'Who. are? you-today'
        target = ['who','are','you','today']
        self.assertEquals(ut.canon_token_sentence(input),target)



if __name__ == '__main__':
   unittest.main()
