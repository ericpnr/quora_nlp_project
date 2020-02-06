def embedding_extraction(filename,embedding_length):
    """
    PURPOSE: Extracting embeddings and a token to id coorespondance dictionary

    ARGS:
    filename        (str) file where the space seperated embeddings are found

    RETURNS:
    embeddings              (list(list(int))) embedding vectors
    token_to_row_number     (dict) of token id number key value pairs
    """
    from collections import defaultdict
    embeddings = []
    running_count = 0
    token_to_row_number = defaultdict(int)

    for line in open(filename):
        full_vector = line.split(' ')
        vector_float = list(map(float,full_vector[1:]))[:embedding_length]
        embeddings.append(vector_float)
        token_to_row_number[full_vector[0]] = running_count
        running_count+=1

    token_to_row_number['UNK'] = running_count

    return embeddings, token_to_row_number


def token_sequence_to_id_sequence(sequences,token_to_row_number,unknown_token,max_sequence_length=100):
    """
    PURPOSE: Convert sequences of token sequences to a sequence of seqeunces of
             corresponding id numbers in token_to_row_number dictionary.

    ARGS:
    sequences               (list(list(str)))
    token_to_row_number     (dict) of token id number key value pairs
    unknown_token           (str) the token in token_to_row_number for the unknown token

    RETURNS:
    id_sequences            (list(list(int)) list of list of id numbers
    """
    unknown_id = token_to_row_number.get(unknown_token)
    id_sequences = []

    for sequence in sequences:
        id_sequence = []
        for token in sequence:
            id_sequence.append(token_to_row_number.get(token,unknown_id))
        if len(id_sequence) >= max_sequence_length:
            id_sequence = id_sequence[:max_sequence_length]
        else:
            id_sequence = id_sequence + [unknown_id]*(max_sequence_length-len(id_sequence))
        id_sequences.append(id_sequence)

    return id_sequences
