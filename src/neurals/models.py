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
from keras.layers import Embedding, Dense, Input, Flatten, LeakyReLU, Dropout, RepeatVector
from keras.layers import Conv1D, MaxPooling1D, LSTM, GRU, Layer, BatchNormalization, Reshape
from keras.layers import Concatenate, GlobalAveragePooling1D, TimeDistributed, Lambda
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Average, CuDNNGRU, CuDNNLSTM
from keras.regularizers import l2
from keras.constraints import MinMaxNorm, MaxNorm
from keras.models import Model, clone_model, load_model
from keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import Adam
from keras.utils import Sequence, plot_model
from keras import backend as K
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


class SentimentFFNN(SentimentModel):
    """SentimentFFNN : Feed Forward Neural Network for sentiment classfication

    Sentiment classification model. 
    Can be used exactly like a regular keras model:
    model = SentimentFFNN(..)
    model.compile(..)
    model.fit(..)

    Args:
        num_classes (Int): 
        num_inputs (Int):
        vocab_size (Int):
        layer_sizes (List[Int]):
        embedding_dim (Int): 
    """

    def __init__(self, num_classes, num_inputs, vocab_size, layer_sizes, embedding_dim):
        self.name='FeedForwardNeuralNetwork'
        assert type(num_classes) == int
        assert type(num_inputs) == int
        assert isinstance(layer_sizes, list)
        assert isinstance(embedding_dim, (int, type(None)))
        assert type(vocab_size) == int
        self.num_inputs = num_inputs
        self.num_classes = num_classes
        self.embed_dim = embedding_dim
        self.vocab_size = vocab_size
        self.layer_sizes = layer_sizes
        self.model = self._construct()

    def _construct(self):
        """_construct : Function for constructing the layers in the model

        Function that constructs the layers that is 
        used in the Feed Forward Neural Network model.
        """
        print(f'Constructing FFNN with {len(self.layer_sizes)} layer(s)....')
        # Creating the Input layer
        self.in_layer = Input(shape=(self.num_inputs, ), dtype='int32', name='InputLayer')

        # Creating the embedding layer
        embed = np.load('../data_phase3_partII/embedding.npy')
        self.embed = Embedding(self.vocab_size, self.embed_dim, weights=[embed], name='EmbeddingLayer')(self.in_layer)

        # Creating Dense and Dropout layers
        self.layer0 = self.embed
        for layer, size in enumerate(self.layer_sizes):
            setattr(self, f'layer{layer}', Dense(size, activation='relu', kernel_regularizer=l2(0.01), name=f'Layer{layer}')(getattr(self, f'drop{layer-1}' if (layer-1) >= 0 else f'layer{layer}')))
            setattr(self, f'drop{layer}', Dropout(0.2, name=f'Dropout{layer}')(getattr(self, f'layer{layer}')))

        # Creating Flatten layer
        self.flat = Flatten(name="FlattenLayer")(getattr(self, f'drop{len(self.layer_sizes)-1}'))

        # Creating output layer
        self.out_layer = Dense(self.num_classes, activation='softmax' if self.num_classes > 1 else 'sigmoid', name="Output")(self.flat)

        # Building model
        return Model(self.in_layer, self.out_layer)


class SentimentCNN(SentimentModel):
    """SentimentCNN : Convolutional Neural Network for sentiment classification

    Class for doing sentiment classification using CNN.
    With this class you can create a custom CNN, 
    but still use it as a keras Model object:
    model = SentimentCNN(..)
    model.compile(..)
    model.fit(..)

    Args:
        num_classes (Int): 
        num_inputs (Int):
        embedding_dim (Int):
        vocab_size (Int):
        layer_sizes (List[(Int, Int, Int)]):
        pool_sizes (List[Int]):
    """

    def __init__(self, num_classes, num_inputs, embedding_dim, vocab_size, layer_sizes, pool_sizes):
        self.name='ConvolutionalNeuralNetwork'
        assert type(num_classes) == int
        assert isinstance(num_inputs, (int, type(None)))
        assert type(layer_sizes) == list
        assert type(embedding_dim) == int
        assert type(vocab_size) == int
        assert type(pool_sizes) == list
        assert len(layer_sizes) == len(pool_sizes)
        self.num_inputs = num_inputs
        self.num_classes = num_classes
        self.embed_dim = embedding_dim
        self.vocab_size = vocab_size
        self.layer_sizes = layer_sizes
        self.pool_sizes = pool_sizes
        self.model = self._construct()

    def _construct(self):
        """_construct : Function for constructing the layers in the model

        Function that constructs the layers that is 
        used in the Convolutional Neural Network model.
        """
        print(f'Constructing CNN with {len(self.layer_sizes)} layer(s)....')

        # Creating Input layer
        self.in_layer = Input(shape=(self.num_inputs, ), dtype='int32', name="InputLayer")

        # Creating embedding layer
        embeddings = np.load('./data/embed_mat.npy')
        self.embed = Embedding(self.vocab_size, self.embed_dim, weights=[embeddings], trainable=True, name="Embedding")(self.in_layer)

        # Creating convolutional layers
        for layer, (filter_, kernel, stride) in enumerate(self.layer_sizes):
            setattr(self, f'conv{layer}', Conv1D(filter_, kernel, strides=stride, activation='relu', kernel_regularizer=l2(0.2), name=f'Conv{layer}')(getattr(self, f'max{layer-1}' if (layer-1) >= 0 else 'embed')))
            setattr(self, f'drop{layer}', Dropout(0.2*(layer+1), name=f'Dropout{layer}')(getattr(self, f'conv{layer}')))
            setattr(self, f'max{layer}', MaxPooling1D(self.pool_sizes[layer], name=f'Max{layer}')(getattr(self, f'drop{layer}')))
        
        # Creating Flattening layer
        self.flat = GlobalAveragePooling1D(name="FlattenLayer")(getattr(self, f'max{len(self.layer_sizes)-1}'))

        # Creating output layer
        self.out_layer = Dense(self.num_classes, activation='softmax' if self.num_classes > 1 else 'sigmoid', name="Output")(self.flat)

        # Building model
        return Model(self.in_layer, self.out_layer)


class SentimentRNN(SentimentModel):
    """SentimentRNN : Recurrent Neural Network for sentiment classification

    Class for doing sentiment classification using RNN.
    With this class you can create a custom RNN, 
    but still use it as a keras Model object:
    model = SentimentRNN(..)
    model.compile(..)
    model.fit(..)

    Args:
        num_classes (Int): 
        num_inputs (Int):
        embedding_dim (Int):
        vocab_size (Int):
        layer_sizes (List[Int]):
        layer_type(LSTM|GRU, optional): Defaults to LSTM.
    """

    def __init__(self, num_classes, num_inputs, embedding_dim, vocab_size, layer_sizes, layer_type='LSTM'):
        self.name = 'RecurrentNeuralNetwork' + layer_type
        assert type(num_classes) == int
        assert isinstance(num_inputs, (int, type(None)))
        assert type(layer_sizes) == list
        assert type(embedding_dim) == int
        assert type(vocab_size) == int
        self.num_inputs = num_inputs
        self.num_classes = num_classes
        self.embed_dim = embedding_dim
        self.vocab_size = vocab_size
        self.layer_sizes = layer_sizes
        self.cell_type = layer_type
        if layer_type == 'LSTM':
            self.layer_type = LSTM
        else:
            self.layer_type = GRU
        self.model = self._construct()

    def _construct(self):
        """_construct : Function for constructing the layers in the model

        Function that constructs the layers that is 
        used in the Recurrent Neural Network model.
        """
        print(f'Constructing RNN with {len(self.layer_sizes)} layer(s)....')

        # Creating Input layer
        self.in_layer = Input(shape=(self.num_inputs, ), dtype='int32', name="InputLayer")

        # Creating Embedding layer
        self.embed = Embedding(self.vocab_size, self.embed_dim, name="EmbeddingLayer")(self.in_layer)

        # Creating Recurrent layers
        for layer, size in enumerate(self.layer_sizes):
            if layer == len(self.layer_sizes) - 1:
                setattr(self, f'layer{layer}', self.layer_type(size, kernel_regularizer=l2(0.1), name=f'Layer{layer}')(getattr(self, f'layer{layer-1}' if (layer-1) >= 0 else 'embed')))
            else:
                setattr(self, f'layer{layer}', 
                    self.layer_type(size, return_sequences=True, kernel_regularizer=l2(0.1), name=f'Layer{layer}')(getattr(self, f'layer{layer-1}' if (layer-1) >= 0 else 'embed'))
                )
        
        # Creating output layer
        self.out_layer = Dense(self.num_classes, activation='softmax' if self.num_classes > 1 else 'sigmoid', name="Output")(getattr(self, f'layer{len(self.layer_sizes)-1}'))

        # Building model instance
        return Model(self.in_layer, self.out_layer)


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

    def _repeat_vector(self, args):
        vec_to_repeat = Reshape((-1,))(args[0])
        sequence_layer = args[1]
        return RepeatVector(K.shape(sequence_layer)[1])(vec_to_repeat)

class SentimentCombinedModel(SentimentModel):
    """SentimentCombinedModel [summary]
        
    Args:
        num_classes ([type]): Number of Output neurons
        params ([type]): Dictionary containing: 
            - rnn_layer_type: (LSTM|GRU)
            - vocab_size
            - embed_dim
            - rnn_sizes: List of RNN Neurons
            - cnn_filters: List of filter sizes
            - cnn_kernels: List of kernel sizes (Int or tuples)
            - cnn_strides: List of stride sizes (Ints or tuples)
            - max_pool_sizes: List of pooling sizes (Ints or tuples)
    
    Raises:
        ValueError: [description]
    """
    def __init__(self, num_classes, params):
        self.name = "SentimentCombinedModel"
        self.num_classes = num_classes
        self.params = params
        if self.params['rnn_layer_type'] == 'LSTM':
            self.layer_type = LSTM
        elif self.params['rnn_layer_type'] == 'GRU':
            self.layer_type = GRU
        else:
            raise ValueError(f'Please specify layer type as either GRU or LSTM, and not {self.params["rnn_layer_type"]}')
        self.model = self._construct()

    def _construct(self):
        # Creating Input Layer
        self.sentence_input = Input(shape=(None, None), dtype='int32', name="SentenceInput")

        # Embedding Layer
        embeddings = np.load('./data/embed_mat.npy')
        self.embedding = TimeDistributed(
            Embedding(
                self.params['vocab_size'], 
                self.params['embed_dim'], 
                weights=[embeddings], 
                trainable=False,
                name="SentenceEmbedding"
            ),
            name="ReviewEmbedding",
            trainable=False
        )(self.sentence_input)

        # RNN Layers
        for layer, size in enumerate(self.params['rnn_sizes']):
            if layer == len(self.params['rnn_sizes']) - 1:
                setattr(self, f'rnnLayer{layer}', TimeDistributed(self.layer_type(size), name=f'RNNLayer{layer}')(getattr(self, f'rnnLayer{layer-1}' if (layer-1) >= 0 else 'embedding')))
            else:
                setattr(self, f'rnnLayer{layer}', TimeDistributed(self.layer_type(size, return_sequences=True), name=f'RNNLayer{layer}')(getattr(self, f'rnnLayer{layer-1}' if (layer-1) >= 0 else 'embedding')))
        
        # Flatten RNN Output
        self.rnn_flatten = GlobalAveragePooling1D(name='RNNFlatten')(getattr(self, f'rnnLayer{len(self.params["rnn_sizes"]) - 1}'))
        # Compute RNN Probs
        self.rnn_output = Dense(self.num_classes, activation='softmax' if self.num_classes > 1 else 'sigmoid', name="RNNOutput")(self.rnn_flatten)

        # CNN Layers
        for layer, filters in enumerate(self.params['cnn_filters']):
            setattr(self, f'cnnLayer{layer}', Conv2D(filters, self.params['cnn_strides'][layer], name=f'CNNLayer{layer}')(getattr(self, f'maxPoolLayer{layer-1}' if (layer-1) >= 0 else 'embedding')))
            setattr(self, f'maxPoolLayer{layer}', MaxPooling2D(self.params['max_pool_sizes'][layer], name=f'MaxPoolLayer{layer}')(getattr(self, f'cnnLayer{layer}')))

        # Flatten CNN Output
        self.cnn_flatten = GlobalAveragePooling2D(name='CNNFlatten')(getattr(self, f'maxPoolLayer{len(self.params["max_pool_sizes"])-1}'))
        # Compute CNN Probs
        self.cnn_output = Dense(self.num_classes, activation='softmax' if self.num_classes > 1 else 'sigmoid', name="CNNOutput")(self.cnn_flatten)

        self.average_output = Average(name="AverageOutput")([self.rnn_output, self.cnn_output])

        return Model(self.sentence_input, self.average_output)
