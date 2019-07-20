# Built-in libraries
import json
import os
from glob import glob
from threading import Thread
from time import sleep
from copy import copy
from random import choice
from abc import ABC, abstractmethod

# Third-party libraries
from tensorflow.keras.layers import Embedding, Dense, Input, Flatten, LeakyReLU, Dropout, RepeatVector
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, GRU, Layer, BatchNormalization, Reshape
from tensorflow.keras.layers import Concatenate, GlobalAveragePooling1D, TimeDistributed, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Average
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import MinMaxNorm, MaxNorm
from tensorflow.keras.models import Model, clone_model, load_model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence, plot_model
from tensorflow.keras import backend as K
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from colorama import Fore
from talos.metrics.keras_metrics import fmeasure_acc

# Custom libraries

BERT_OUT_DIM = 768
COLORS = [Fore.RED, Fore.BLUE, Fore.GREEN,
          Fore.YELLOW, Fore.MAGENTA, Fore.CYAN]


class SentimentModel(ABC):
    def __init__(self):
        self.model = None
        self.name = None

    @abstractmethod
    def _construct(self):
        pass

    def compile(self, optimizer, loss, metrics, options=None):
        self.model.compile(optimizer, loss, metrics, options=options)

    def predict_on_batch(self, x):
        return self.model.predict_on_batch(x)

    def train_on_batch(self, x, y):
        return self.model.train_on_batch(x, y)

    def to_json(self):
        return self.model.to_json()

    def fit(self, x, y, validation_data, epochs, batch_size, verbose):
        return self.model.fit(
            x=x, 
            y=y, 
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )

    def fit_generator(self, generator, validation_data, epochs, callbacks=None, verbose=1, use_multiprocessing=False, workers=1, max_queue_size=2):
        return self.model.fit_generator(
            generator=generator,
            validation_data=validation_data,
            epochs=epochs,
            use_multiprocessing=use_multiprocessing,
            workers=workers,
            verbose=verbose,
            callbacks=callbacks,
            max_queue_size=max_queue_size
        )

    def evaluate_generator(self, generator, steps=None, max_queue_size=2, workers=1, use_multiprocessing=False, verbose=0):
        return self.model.evaluate_generator(generator, 
                steps=steps, 
                # callbacks=callbacks, 
                max_queue_size=max_queue_size, 
                workers=workers, 
                use_multiprocessing=use_multiprocessing, 
                verbose=verbose)
 
    def predict_generator(self, generator, steps=None, max_queue_size=2, workers=1, use_multiprocessing=False, verbose=0):
        return self.model.predict_generator(generator, 
                steps=steps, 
                # callbacks=callbacks, 
                max_queue_size=max_queue_size, 
                workers=workers, 
                use_multiprocessing=use_multiprocessing, 
                verbose=verbose)

    def summary(self, line_length=100):
        self.model.summary(line_length=line_length)

    def plot(self):
        plot_model(self.model, to_file=f'{self.name}.png')

    def save(self, path):
        self.model.save(path)

    def get_weights(self):
        return self.model.get_weights()

    def load(self, path):
        self.model = load_model(path, custom_objects={'fmeasure_acc': fmeasure_acc})


class SentimentSentenceRNN(SentimentModel):
    """SentimentSentenceRNN : Sentence level sentiment classifier using RNN cells

    Class for doing sentiment classification on sentence level using RNN.
    With this class you can create a custom RNN, 
    but still use it as a keras Model object:
    model = SentimentSentenceRNN(..)
    model.compile(..)
    model.fit(..)
    
    Args:
            num_classes (Int): [description]
            params ([type]): Dictionary containing: 
            - rnn_layer_type: (LSTM|GRU)
            - vocab_size
            - embed_dim
            - num_topics
            - topic_embed
            - rnn_sizes: List of RNN Neurons
            - dense_sizes: List of Dense layer sizes

    Returns:
        SentimentSentenceRNN object: Returns SentimentSentenceRNN object, which is a subclass of keras Model object.
    """
    def __init__(self, num_classes, params):
        self.name = "SentimentSentenceRNN"
        self.num_classes = num_classes
        self.params = params
        if self.params['layer_type'] == 'LSTM':
            self.layer_type = LSTM
        elif self.params['layer_type'] == 'GRU':
            self.layer_type = GRU
        else:
            raise ValueError(f'Please specify layer type as either GRU or LSTM, and not {self.params["layer_type"]}')
        self.layer_type_text = self.params['layer_type']
        self.model = self._construct()


    def _construct(self):
        """_construct : Function for constructing the layers in the model

        Function that constructs the layers that is 
        used in the Sentence Based Recurrent Neural Network model.
        """
        print(f'Constructing Sentence RNN with {len(self.params["rnn_layer_sizes"])} & {len(self.params["dense_sizes"])} layer(s)....')

        self.sentence_input = Input(shape=(None,), dtype='int32', name="ReviewInput")
        self.topic_input = Input(shape=(1,), dtype='int32', name="TopicInput")

        # Creating Embedding layers
        embeddings = np.load(self.params['embedding'])
        self.sentence_embedding = Embedding(self.params['vocab_size'], self.params['sent_embed'], weights=[embeddings], trainable=False, name="ReviewEmbedding")(self.sentence_input)
        self.topic_embedding = Embedding(self.params['num_topics'], self.params['topic_embed'], name="TopicEmbedding")(self.topic_input)
        
        # Creating a Repetition layer
        self.flat_layer = Flatten(name="TopicEmbedFlatten")(self.topic_embedding)

        # Creating the sentence feature extraction
        for layer, size in enumerate(self.params["rnn_layer_sizes"]):
            if layer == len(self.params["rnn_layer_sizes"]) - 1:
                setattr(self, f'sentLayer{layer}', self.layer_type(size, recurrent_regularizer=l2(0.01), name=f'SentLayer{layer}')(getattr(self, f'sentLayer{layer-1}' if (layer-1) >= 0 else 'sentence_embedding')))
            else:
                setattr(self, f'sentLayer{layer}', self.layer_type(size, recurrent_regularizer=l2(0.01), return_sequences=True, name=f'SentLayer{layer}')(getattr(self, f'sentLayer{layer-1}' if (layer-1) >= 0 else 'sentence_embedding')))
        
        # Creating the Concatenation layer
        self.concat = Concatenate(axis=-1, name="ConcatLayer")([getattr(self, f'sentLayer{len(self.params["rnn_layer_sizes"])-1}'), self.flat_layer])

        # Dense layers
        for layer, size in enumerate(self.params['dense_sizes']):
            if self.params['ablate']:
                setattr(self, f'denseLayer{layer}', Dense(size, activation='relu')(getattr(self, f'dropoutLayer{layer-1}' if (layer-1) >= 0 else f'sentLayer{len(self.params["rnn_layer_sizes"])-1}')))
            else:
                setattr(self, f'denseLayer{layer}', Dense(size, activation='relu')(getattr(self, f'dropoutLayer{layer-1}' if (layer-1) >= 0 else 'concat')))
            setattr(self, f'dropoutLayer{layer}', Dropout(0.2)(getattr(self, f'denseLayer{layer}')))

        # Creating output layer
        self.out_layer = Dense(self.num_classes, activation='softmax' if self.num_classes > 1 else 'sigmoid', name="Output")(getattr(self, f'dropoutLayer{len(self.params["dense_sizes"])-1}'))

        # Building model instance
        return Model([self.sentence_input, self.topic_input], self.out_layer)