# Built-in libraries
import json
import os
from glob import glob
from threading import Thread
from time import sleep
from copy import copy
from random import choice, random

# Third-party libraries
from keras.layers import Embedding, Dense, Input, Flatten, LeakyReLU, Dropout, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, LSTM, GRU, Layer, GlobalAveragePooling1D
from keras.regularizers import l2
from keras.constraints import MinMaxNorm, MaxNorm
from keras.models import Model, clone_model
from keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import Adam
from keras.utils import Sequence, plot_model, to_categorical
from keras import backend as K
import numpy as np
# from bert_serving.client import BertClient
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from colorama import Fore

# Custom libraries
from ..utils import mapcount  # , start_bert, stop_bert

BERT_OUT_DIM = 768
COLORS = [Fore.RED, Fore.BLUE, Fore.GREEN,
          Fore.YELLOW, Fore.MAGENTA, Fore.CYAN]


class SentenceSentimentTopicGenerator(Sequence):
    """SentenceSentimentTopicGenerator : Generator for training advanced neural models

    Args:
        filename (str): File with preprocessed and tokenized data
        vocab (str): File in where the vocab lives
        seq_len (int, optional): Defaults to None. Maximum sequence length. Leave it to None if you want it to be variable 
        batch_size (int, optional): Defaults to 32. Batch size when running
        classes (list, optional): Defaults to ['negative', 'positive']. Class labels to convert to ids 
        bert (bool, optional): Defaults to False. Whether you use bert or not. 
        validation (bool, optional): Defaults to False. If the generator is pulling validation or training data. 
    """

    def __init__(self, filename, vocab, review_len, seq_len=None, batch_size=32, classes=['negative', 'positive'], ablate=False):
        assert isinstance(filename, str)
        assert isinstance(vocab, str)
        assert isinstance(batch_size, int)
        assert isinstance(seq_len, (int, type(None)))
        self.seq_len = seq_len
        self.review_len = review_len
        self.data_file = filename
        self.batch_size = batch_size
        self.labels = classes
        self.vocab = json.load(open(vocab, 'r'))
        self.index = 0
        self.ablate = ablate
        self.data = open(self.data_file, 'r')

    def __len__(self):
        # Return the number of batches in an epoch.
        return int(np.ceil(mapcount(self.data_file)/self.batch_size))

    def __getitem__(self, index):
        X, y = self._get_data(index)
        return X, y

    def on_epoch_end(self):
        # Re-read the data file on epoch end
        self.data.close()
        self.data = open(self.data_file, 'r')
        self.index = 0

    def _get_data(self, index):
        X, y, topic = [], [], []
        # print("Worker index", index)
        iters = min(self.batch_size, mapcount(
            self.data_file)-(self.index*self.batch_size))
        self.index += 1
        for _ in range(iters):
            line = json.loads(next(self.data))
            l = []
            for sent in line['review']:
                s = []
                for word in sent:
                    try:
                        s.append(self.vocab[word])
                    except KeyError:
                        s.append(self.vocab["<UNK>"])
                l.extend(s)

            X.append(l)
            y.append(self.labels.index(line['y']))
            topic.append(int(line['topic']))

        X = np.array(self._pad_reviews(X)).reshape((iters, -1))

        if not self.ablate:
            return [np.array(X), np.array(topic)], y
        else:
            return np.array(X), y

    def _pad_sent(self, text, max_len):
        sent_pad = []
        diff = max_len - len(text)
        if diff > 0:
            sent_pad.append(text + ([0] * diff))
        elif diff < 0:
            sent_pad.append(text[:max_len])
        else:
            sent_pad.append(text)
        return sent_pad

    def _pad_reviews(self, reviews):
        try:
            max_sent_len = len(max(reviews, key=len))
        except ValueError as e:
            print(e)
            print(f"At this point our index is {self.index}")
            exit()
        if max_sent_len > self.review_len:
            max_sent_len = self.review_len
        reviews = [self._pad_sent(review, max_sent_len) for review in reviews]
        return reviews
