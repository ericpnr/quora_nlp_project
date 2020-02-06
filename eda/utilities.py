import re

def canonicalize_digits(word):
    """
    PURPOSE: To convert all digits within a word to 'DG' token

    ARGS:
        word   'str' string to be canonicalized

    RETURNS:
        work    'str' canonicalized word
    """
    if any([c.isalpha() for c in word]): return word
    word = re.sub("\d", "DG", word)
    if word.startswith("DG"):
        word = word.replace(",", "") # remove thousands separator
    return word


def canonicalize_word(word, wordset=None, digits=True):
    """
    PURPOSE:
    PURPOSE: To convert all digits within a word to 'DG' token

    ARGS:
        word   'str' string to be canonicalized

    RETURNS:
        word    'str' canonicalized word
    """
    word = word.lower()
    if digits:
        if (wordset != None) and (word in wordset): return word
        word = canonicalize_digits(word) # try to canonicalize numbers
    if (wordset == None) or (word in wordset):
        return word
    else:
        return constants.UNK_TOKEN


def canonicalize_words(words, **kw):
    """
    PURPOSE: Canonicalize a list of words

    ARGS:
        words  (list(str)) list of words

    RETURNS:
        list of canonicalized words
    """
    return [canonicalize_word(word, **kw) for word in words]


def canonicalize_sentence(sentence, **kw):
    """
    PURPOSE: Canonlicalize a sentence

    ARGS:
        sentence   (str) sentence to be canonicalized

    RETURNS:
        sentence   (str) canonicalized sentence
    """
    sentence = sentence.lower()
    sentence = re.sub("(?<=\S)[\.\/\?\(\)\[-](?=\S)"," ",sentence)
    sentence = re.sub("(?<=\S)[\.\/\?\(\)\[](?=\s)","",sentence)
    sentence = re.sub("(?<=\s)[\.\/\?\(\)\[](?=\S)","",sentence)
    sentence = re.sub("(?<=\s)[\.\/\?\(\)\[](?=\s)","",sentence)
    sentence = re.sub("['\.\/\?\(\)\[](?=$)","",sentence)
    sentence = re.sub("[^a-zA-Z\d\s]","",sentence)
    sentence = re.sub("\s{2,}"," ",sentence)
    return sentence


def canon_token_sentence(sentence,**kw):
    """
    PURPOSE: Canonicalize and tokenize a sentence

    ARGS:
        sentence (str) sentence for canonicalization

    RETURNS:
        canon_token   (list) tokenized sentence
    """
    canon_sentence = canonicalize_sentence(sentence)
    canon_token = canonicalize_words(canon_sentence.split(' '))
    return canon_token


def flatten_sort_listx2_tuple(tpl_list,sort_index=0,ascending='TRUE'):
    """
    PURPOSE: Flatten and sort a list of tuples.

    ARGS:
        tpl_list    (list(tpl)) list of tuples
        sort_index  (int) which element of the tuple to sort by
        ascending   (bool) sorting type

    RETURNS:
        tpl_list   (list(tpl)) sorted list of tuples
    """
    tpl_list = [ item for sublist in tpl_list for item in sublist]
    if ascending:
        tpl_list = sorted(tpl_list,key = lambda x: -x[sort_index])
    else:
        tpl_list = sorted(tpl_list,key = lambda x: x[sort_index])
    return tpl_list
