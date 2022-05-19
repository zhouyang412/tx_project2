#!/usr/bin/env python
# -*- coding: utf-8 -*-


from collections import Counter

import numpy as np


class Vocab(object):
    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 3

    def __init__(self, init_path=None):
        self.word2index = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.index2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.word2count = Counter()
        self.reserved = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        #         self.index2word = self.reserved[:]
        self.embeddings = None
        self.idx = 4

        if init_path is not None:
            with open(init_path, 'r')as f:
                words = f.readlines()[5:]
                words = [w.strip() for w in words]
            self.add_words(words)
        
        
    def add_words(self, words):
        """Add a new token to the vocab and do mapping between word and index.

        Args:
            words (list): The list of tokens to be added.
        """
        ###########################################
        #          TODO: module 1 task 1          #
        ###########################################
        
        for word in words:
            self.word2index[word] = self.word2index.get(word, self.idx)
            self.index2word[self.idx] = word
            self.reserved.append(word)
            self.word2count.update(word)
            self.idx += 1
            

    def load_embeddings(self, file_path: str, dtype=np.float32) -> int:
        """Load embedding word vector.

        Args:
            file_path (str): The file path of word vector to load.
            dtype (numpy dtype, optional): Defaults to np.float32.

        Returns:
            int: Number of embedded tokens.
        """
        num_embeddings = 0
        vocab_size = len(self)
        with open(file_path, 'rb') as f:
            for line in f:
                line = line.split()
                word = line[0].decode('utf-8')
                idx = self.word2index.get(word)
                # Check whether the token is in the vocab.
                # 如果在词表中就更新embeddings
                if idx is not None:
                    vec = np.array(line[1:], dtype=dtype)
                    if self.embeddings is None:
                        # Get embedding dimension.
                        n_dims = len(vec)
                        # Initialize word vectors.
                        self.embeddings = np.random.normal(
                            np.zeros((vocab_size, n_dims))).astype(dtype)
                        pad_idx = self.word2index['<PAD>']
                        self.embeddings[pad_idx] = np.zeros(n_dims)
                    # 更新embeddings，key为对应的wordidx，value为词向量    
                    # self.embeddings --> {word_id1:word1_vec, word_idx2:word2_vec...}
                    self.embeddings[idx] = vec
                    num_embeddings += 1
        return num_embeddings

    def __getitem__(self, item):
        # __getitem__返回item对应的value，即输入词返回idx或者输入idx返回词
        if type(item) is int:
            return self.index2word[item]
        return self.word2index.get(item, self.UNK)

    def __len__(self):
        return len(self.index2word)

    def size(self):
        """Returns the total size of the vocabulary"""
        return len(self.index2word)
