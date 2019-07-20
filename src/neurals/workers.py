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


class SentimentWorker:
    """ SentimentWorker : Worker for handling training of Sentiment Classification models.

    This is a wrapper for handling compilation and training of Sentiment model.
    This takes a keras model (One of the above) and some data in either ndarray or generator style.
    Then you can use the built in methods for training, compiling and validation of model.

    Args:
        model (keras.Model): [description]
        data (tuple[ndarray, ndarray], optional): Defaults to None. Tuple of X, y train data
        data_val (tuple[ndarray, ndarray], optional): Defaults to None. Tuple of X, y validation data
        generator (SentimentGenerator, optional): Defaults to None. Generator for train data
        generator_val (SentimentGenerator, optional): Defaults to None. Generator for validation data
    """

    def __init__(self, model, data=None, data_val=None, generator=None, generator_val=None, save=None, load=None):
        assert isinstance(model, Model)
        if not data:
            X, y = None, None
        else:
            X, y = data
        if not data_val:
            X_val, y_val = None, None
        else:
            X_val, y_val = data_val
        assert isinstance(generator, (SentimentGenerator,
                                      type(None), CrossValGenerator))
        assert isinstance(generator_val, (SentimentGenerator, type(None)))
        if isinstance(generator, CrossValGenerator):
            self.model = model
        else:
            inp = Input(shape=(None, ), dtype='int32')
            self.model = Model(inputs=inp, outputs=model(inp))
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val
        self.generator = generator
        self.generator_val = generator_val
        self.step = 0
        self.save = save
        if load != None:
            self.model.load_weights(load)

    def compile_model(self, optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy']):
        """compile_model Compile the supplied keras model.

        Compiles the supplied keras model using the specified parameters. 
        This function is just a wrapper for the original keras.models.Model function

        Args:
            optimizer (keras.optimizers, optional): Defaults to Adam(0.001). [description]
            loss (str, optional): Defaults to 'binary_crossentropy'. [description]
            metrics (list, optional): Defaults to ['accuracy']. [description]
        """
        self.opt = optimizer
        self.loss_fn = loss
        self.model.compile(optimizer, loss, metrics)

    def train_model(self, epochs, batch_size=128, tensorboard=None, save_file='./data/viz.txt'):
        """train_model Train the supplied model.

        Train the keras model using the data supplied in the beginning. 
        This data could either be a generator or two numpy arrays. 

        Args:
            epochs (Int): Number of epochs to train for
            batch_size (Int, optional): Defaults to 128. The batch size to use when training on ndarrays or doing cross_val.
            cross_val (bool, optional): Defaults to False. Wether or not to do Cross Validation.
            tensorboard (keras.callbacks.TensorBoard, optional): Defaults to None. Callback from keras utilizing tensorboard.

        Raises:
            ValueError: Raises an error if no data was supplied in the beginning.
        """
        if isinstance(self.X, np.ndarray) and isinstance(self.y, np.ndarray):
            # If data is np arrays run standard fitting function
            self.model.fit(self.X, self.y, epochs=epochs,
                           batch_size=batch_size, callbacks=tensorboard)
            self.model.summary()
            if self.save != None:
                self.model.save(self.save)
        elif isinstance(self.generator, SentimentGenerator):
            # Run the training using the generator
            callbacks = None
            if tensorboard != None:
                files = len(glob('./logs/*'))
                tb = tensorboard(log_dir=f'./logs/sentiment_logs{files+1}',
                                 batch_size=self.generator.batch_size,
                                 write_graph=True)
                callbacks = [tb]

            _ = self.model.fit_generator(
                generator=self.generator,
                validation_data=self.generator_val,
                epochs=epochs,
                use_multiprocessing=True,
                workers=4,
                callbacks=callbacks
            )
            
            # TODO: Can save Hist here?

            if self.save != None:
                self.model.save(self.save)
        elif isinstance(self.generator, CrossValGenerator):
            loss, acc = (
                [[] for _ in range(self.generator.n_splits)], 
                [[] for _ in range(self.generator.n_splits)]
            )
            preds = [np.array([]) for _ in range(self.generator.n_splits)]
            true = [np.array([]) for _ in range(self.generator.n_splits)]
            name = self.model.name
            models = [self.model.copy(self.opt, self.loss_fn)
                      for _ in range(self.generator.n_splits)]
            del self.model
            for i, model in enumerate(models):
                self.step = 0
                train, test = self.generator[i]
                if tensorboard != None:
                    tb = tensorboard(log_dir=f'./logs/CrossVal/cross_val_{name}_{i}_e_{epochs}',
                                     batch_size=self.generator.batch_size,
                                     write_graph=True)
                    tb.set_model(model)
                for epoch in range(epochs):
                    pbar = tqdm(range(len(train[0])), desc=f'FOLD {i}: Epoch {epoch+1}/{epochs}',
                                bar_format="{l_bar}%s{bar}%s{r_bar}" % (
                                    choice(COLORS), Fore.RESET),
                                unit='batches')
                    for step in pbar:
                        X, y = train[0][step], train[1][step]
                        X = self.pad_seq(X)
                        metrics = model.train_on_batch(X, y)
                        l, a = metrics[0], metrics[1]
                        pbar.set_postfix_str(
                            f'Loss: {l:.2f} - Accuracy: {a:.2f}')
                        if tensorboard != None:
                            tb.on_epoch_end(
                                self.step, self.create_metrics(metrics))
                            self.step += 1

                    metrics_val = [0, 0]
                    for step in range(len(test[0])):
                        X_val, y_val = test[0][step], test[1][step]
                        X_val = self.pad_seq(X_val)
                        val = model.test_on_batch(X_val, y_val)
                        metrics_val[0] += val[0]
                        metrics_val[1] += val[1]
                    metrics_val[0] /= len(test[0])
                    metrics_val[1] /= len(test[0])
                    loss[i].append(metrics_val[0])
                    acc[i].append(metrics_val[1])
                    print(
                        f'Loss: {metrics[0]:.2f} - Accuracy: {metrics[1]:.2f} - Loss Val: {metrics_val[0]:.2f} - Accuracy Val: {metrics_val[1]:.2f}')
                    if tensorboard != None:
                        tb.on_epoch_end(epoch, self.create_metrics(
                            metrics_val, test=True))
                for step in range(len(test[0])):
                    preds[i] = np.append(preds[i], model.predict_on_batch(
                        self.pad_seq(test[0][step])))
                    true[i] = np.append(true[i], test[1][step])
                preds[i] = preds[i].tolist()
                true[i] = true[i].tolist()
            with open(save_file, 'a+') as viz:
                result = {"name": name, "neural": True, "cross_val": True,
                          "loss": loss, "accuracy": acc, "predictions": preds, "true": true}
                json.dump(result, viz)
                viz.write("\n")
        else:
            raise ValueError(
                "Please supply either a SentimentGenerator, CrossValGenerator or two ndarrays")

    def pad_seq(self, data):
        max_seq_len = len(
            max(data, key=len)) if not self.generator.use_seq_len else self.generator.seq_len
        X_pad = []
        for text in data:
            diff = max_seq_len - len(text)
            if diff > 0:
                X_pad.append(text + ([0] * diff))
            elif diff < 0:
                X_pad.append(text[:max_seq_len])
            else:
                X_pad.append(text)
        return np.array(X_pad)

    def create_metrics(self, metrics, test=False):
        """create_metrics

        Create dict of metrics with labels

        Args:
            metrics (list): List of metrics
            test (bool, optional): Defaults to False. Whether or not it is metrics performed in test.

        Returns:
            dict: Dict with {metric_type: metric_val}
        """
        if test:
            labels = ['Validation Loss', 'Validation Accuracy']
        else:
            labels = ['Loss', 'Accuracy']
        d = dict()
        for l, m in zip(labels, metrics):
            d[l] = m
        return d


class SentimentGenerator(Sequence):
    """SentimentGenerator : Generator for training neural models

    [description]

    Args:
        filename (str): File with preprocessed and tokenized data
        vocab (str): File in where the vocab lives
        seq_len (int, optional): Defaults to None. Maximum sequence length. Leave it to None if you want it to be variable 
        batch_size (int, optional): Defaults to 32. Batch size when running
        classes (list, optional): Defaults to ['negative', 'positive']. Class labels to convert to ids 
        bert (bool, optional): Defaults to False. Whether you use bert or not. 
        validation (bool, optional): Defaults to False. If the generator is pulling validation or training data. 
    """

    def __init__(self, filename, vocab, seq_len=None, batch_size=32, classes=['negative', 'positive', 'hidden'], validation=False):
        assert isinstance(filename, str)
        assert isinstance(vocab, str)
        assert isinstance(batch_size, int)
        assert isinstance(seq_len, (int, type(None)))
        self.seq_len = seq_len
        self.data_file = filename
        self.batch_size = batch_size
        self.labels = classes
        self.vocab = json.load(open(vocab, 'r'))
        self.index = 0
        self.data = None
        self.on_epoch_end()
        self.val = validation

    def __len__(self):
        # Return the number of batches in an epoch.
        return int(np.ceil(mapcount(self.data_file)/self.batch_size))

    def __getitem__(self, index):
        if self.val:
            X, y, ids = self._get_data(index)
            return X, y, ids
        else:
            X, y = self._get_data(index)
            return X, y

    def on_epoch_end(self):
        # Re-read the data file on epoch end
        self.data = open(self.data_file, 'r')

    def _get_data(self, index):
        X, y = [], []
        if self.val:
            ids = []
        iters = min(self.batch_size, mapcount(self.data_file)-(index*self.batch_size))
        for _ in range(iters):
            line = json.loads(next(self.data))
            l = []
            for word in line['sentence']:
                try:
                    l.append(self.vocab[word])
                except KeyError:
                    l.append(self.vocab["<UNK>"])
            X.append(l)
            y.append(self.labels.index(line['y']))
            if self.val:
                ids.append(line['id'])

        max_seq_len = len(max(X, key=len)) if not self.seq_len else self.seq_len
        X_pad = []
        for text in X:
            diff = max_seq_len - len(text)
            if diff > 0:
                X_pad.append(text + ([0] * diff))
            elif diff < 0:
                X_pad.append(text[:max_seq_len])
            else:
                X_pad.append(text)
        if self.val:
            return np.array(X_pad), np.array(y), ids
        else:
            if len(self.labels) > 2:
                return np.array(X_pad), np.array(to_categorical(y, num_classes=len(self.labels)))
            else:
                return np.array(X_pad), np.array(y)


class CrossValGenerator:
    def __init__(self, filename, vocab, n_splits, batch_size=32, classes=['negative', 'positive'], seq_len=None, use_seq_len=False):
        self.data_file = filename
        self.n_splits = n_splits
        self.vocab = json.load(open(vocab, 'r'))
        self.labels = classes
        self.splits = None
        self.batch_size = batch_size
        self.use_seq_len = use_seq_len
        self.seq_len = seq_len
        self._prep_data()

    def _prep_data(self):
        data = [json.loads(line.rstrip())
                for line in open(self.data_file, 'r')]
        self.X, self.y = [], []
        for line in data:
            l = []
            for word in line['text']:
                try:
                    l.append(self.vocab[word])
                except KeyError:
                    l.append(self.vocab["<UNK>"])
            self.X.append(l)
            self.y.append(self.labels.index(line['y']))
        self.X, self.y = np.array(self.X), np.array(self.y)
        self.splits = StratifiedKFold(
            n_splits=self.n_splits).split(self.X, self.y)

    def __getitem__(self, index):
        tr_index, te_index = next(self.splits)
        X_tr, y_tr = self.X[tr_index], self.y[tr_index]
        X_te, y_te = self.X[te_index], self.y[te_index]
        no_tr, no_te = int(
            len(X_tr)/self.batch_size), int(len(X_te)/self.batch_size)
        train = (np.array_split(X_tr, no_tr), np.array_split(y_tr, no_tr))
        test = (np.array_split(X_te, no_te), np.array_split(y_te, no_te))
        return train, test

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
