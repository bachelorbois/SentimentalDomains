# Built-in libraries
import mmap
import json
from random import choice

# Third-party libraries
from keras.models import Model
from sklearn.model_selection import StratifiedKFold
import numpy as np

def sort_out(y, ind):
    '''[summary]
    
    Args:
        y (list): List of "positive" "negative"
        ind (list): List of indices
    '''
    print(len(ind), len(y))
    return list([x for _,x in sorted(zip(ind,y))])

def embed_mat(dim="50d", path="./data/word2int.json"):
    with open(path, 'r') as f:
        word_index = json.load(f)
    embedding_index = {}
    with open(f"./data/glove.6B.{dim}.txt", 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype=np.float32)
            embedding_index[word] = coefs
    
    embedding_matrix = np.zeros((len(word_index) + 1, 50))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def save_embed_mat(f="./data/embed_mat.npy", path="./data/word2int.json"):
    mat = embed_mat(path=path)
    np.save(f, mat)

def merge_vocab():
    groups = ['group_1', 'group_2', 'group_3', 'group_4', 'group_5', 'group_6', 'group_7', 'group_8', 'group_9']
    vocab = {}
    for group in groups:
        vocab.update(json.load(open(f'./data_phase3_partII/{group}/word2int.json')))
    i = 0
    for key, _ in vocab.items():
        vocab[key] = i
        i += 1
    json.dump(vocab, open('./data_phase3_partII/word2int.json', 'w'))

def mapcount(filename):
    with open(filename, "r+") as f:
        return mapcount_fp(f)

def mapcount_fp(fp):
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    readline = buf.readline
    while readline():
        lines += 1
    return lines
