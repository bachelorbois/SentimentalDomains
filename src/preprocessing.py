#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import Counter
from nltk.tokenize import wordpunct_tokenize, RegexpTokenizer
from nltk.corpus import wordnet, stopwords
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
#from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from nltk import pos_tag
#from pattern.en import tag

from io import IOBase
import csv
#from spellchecker import SpellChecker #pip install pyspellchecker
#from symspellpy.symspellpy import SymSpell, Verbosity
import string
import time
import os
import gc
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
import itertools
import json
import re
#from hunspell import HunSpell
gc.collect()
import sys
from tqdm import tqdm 

class Preprocessor:

    """Preprocesses .csv file 
    and creates an output file 
    in the output directory 
    ultilizing multithreading,
    making processing of 
    large file(s) bearable.
    
    The preprocessing includes:
        - tokenization
        - spellchecking
        - stemming
        - excluding of stopwords
        - generation of a vocabulary
    """
    
    def __init__(self, filename, sentence = True, tagging = True, output_dir = '../data/'):
        
        assert type(filename) == str
        assert type(output_dir) == str
        
        self.filename = filename
        self.output_dir = output_dir
        self.sentence = sentence
        self.vocab = Counter()
        self.tagging = tagging
        self.tag = set()
        
        # Constructing STOP_WORDS set
        #stop = ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with']
        stop = list(stopwords.words('english'))
        # print(stop)
        # keep_set = set(re.findall(r'not|no|[a-z]+\'t', ' '.join(stop)))
        # stop = stop - keep_set
        # print(stop)
        self.tags_set = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'MNP', 'MNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'STOP_WORDS']
       #self.stemmer = HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')
        
        self.tokenizer = RegexpTokenizer(r'[^\W\d_]+|[\.]{3}|[:;,()!\"\']{1}|\'ll|\'nt')
        
        self._run()
    

    def _preprocess_data(self, row):
        """Private method that preprocesses one line at a time.
        
        :row: A list consists of elements separated by commas.
        :returns: A tuple of dictionary representing the processed row and a set of vocabulary.

        """
        tokens = self._tokenize(row[0])
        tokens = list(map(lambda x: x.lower(), tokens))
        #tokens = self._spellcheck(tokens)
        #print(tokens)
        #for i, w in enumerate(tokens):
        #    if self.stemmer.spell(w) == False:
        #        suggestions = self.stemmer.suggest(w)
        #        if suggestions:
        #            tokens[i] = suggestions[0]
        #tokens_text = [self._stem(w) for w in tokens if w not in self.stop]
        vocab_set = set(tokens)
        chunk = {'y': row[1].strip(), 'sentence': tokens}
        
        return chunk, vocab_set

    def _tokenize(self, words):
        tokenized = self.tokenizer.tokenize(words)
        return tokenized
        #l = re.findall(r"\w+(?=n't)|n't|\w+(?=')|'\w+|\w+|[()!?\"']+|\.{3}",s, re.IGNORECASE|re.DOTALL)
        #l = re.sub(r'n\'t', 'not', ' '.join(l))
        #l = re.sub(r'^\'\w+|', '', l).split()
        #return l
    
    def save_vocab(self, constraint = 100):
        """ Excludes the words in the vocabulary 
        that has an occurence below 10.
        Map integer to each word and vice versa.
        Adds zero pad token, unknown token to both word2int and int2word.
        Writes 2 corresponding .json files into the output directory.
        :constraint: integer (default 10)
        """
        
        v = {k:v for k, v in self.vocab.items() if v >= constraint}
        v = sorted(v.keys())
        word2int = {}
        int2word = {}
        word2int['<0>'] = 0
        word2int['<UNK>'] = 1
        int2word[0] = '<0>'
        int2word[1] = '<UNK>'

        for i, w in enumerate(v):
           word2int[w] = i+2
           int2word[i+2] = w
        file_pair = [(word2int, 'word2int'), (int2word, 'int2word')]
        for d, v in file_pair:
            with open(self.output_dir + v + '.json', 'w') as output:
                json.dump(d, output)

    
    def _get_next_line(self, filename = None):
        """Yields iterator of the data file.
        If no file argument is supplied, 
        it assumes the initialized file.
        """
        if filename == None:
            filename = self.filename
        
        with open(self.output_dir + filename, 'r') as f:
            if self.tagging == False and self.sentence == True:
                for line in f:
                    yield json.loads(line)
            else:
                reader = csv.reader(f)
                for row in reader:
                    yield row
 


    def _run(self):
        """Runs preprocessing concurrently
        using threadpools, then merge the results.
        Sort the merged result by the text length
        and write to and output file in the output directory.
        """
        f = self._get_next_line()
        #results = []
        with open(self.output_dir + 'output.txt', 'w') as output:
            with ThreadPool(4) as p:
                for i in tqdm(f):
                    joined_result = p.map(self._preprocess_data, (i,))
                    for t, v in joined_result:
                        #print(t)
                        self.vocab.update(v)
                        json.dump(t, output)
                        output.write('\n')
        #data = []
        #with open(self.output_dir + 'output.txt', 'r') as fin:
            #for line in fin:
                #data.append(json.loads(line))    
               
        #data = sorted(data, key = lambda x : len(x['review']))

        #with open(self.output_dir + 'output.txt', 'w') as fout:
            #for line in data:
                #json.dump(line, fout)
                #fout.write('\n')

                    
if __name__ == '__main__':
    if len(sys.argv) == 2:
        prec = Preprocessor(str(sys.argv[1]), tagging = True)
        #prec.save_vocab()
    elif len(sys.argv) == 3:
        prec = Preprocessor(str(sys.argv[1]), sentence = True, tagging = False, output_dir = str(sys.argv[2]) )
        prec.save_vocab()
