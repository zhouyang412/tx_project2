#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import pathlib
from collections import Counter
from typing import Callable

import torch
from torch.utils.data import Dataset

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))
from utils import simple_tokenizer, count_words, sort_batch_by_len, source2ids, abstract2ids
from vocab import Vocab
import config


class PairDataset(object):
    """The class represents source-reference pairs.

    """
    def __init__(self,
                 filename,
                 tokenize: Callable = simple_tokenizer,
                 max_src_len: int = config.max_src_len - 2,
                 max_tgt_len: int = config.max_tgt_len,
                 truncate_src: bool = False,
                 truncate_tgt: bool = False):
        print("Reading dataset %s..." % filename, end=' ', flush=True)
        self.filename = filename
        # 这里存储src和tgt句子对
        self.pairs = []

        with open(filename, 'rt', encoding='utf-8') as f:
            next(f)
            for i, line in enumerate(f):
                # Split the source and reference by the <sep> tag.
                # 根据<sep> 进行切分
                pair = line.strip().split('<sep>')
                # Check whether there is an actual pair.
                if len(pair) != 2:
                    print("Line %d of %s is malformed." % (i, filename))
                    print(line)
                    continue
                # 对数据进行split，simple_tokenizer就是按空格进行切分
                src = tokenize(pair[0])
                # 大于规定长度则进行截取
                if max_src_len and len(src) > max_src_len:
                    if truncate_src:
                        src = src[:max_src_len]
                    else:
                        continue
                tgt = tokenize(pair[1])
                if max_tgt_len and len(tgt) > max_tgt_len:
                    if truncate_tgt:
                        tgt = tgt[:max_tgt_len]
                    else:
                        continue
                self.pairs.append((src, tgt))
        print("%d pairs." % len(self.pairs))

    def build_vocab(self, embed_file: str = None) -> Vocab:
        """Build the vocabulary for the data set.

        Args:
            embed_file (str, optional):
            The file path of the pre-trained embedding word vector.
            Defaults to None.

        Returns:
            vocab.Vocab: The vocab object.
        """
        # word frequency
        # 遍历每一组数据，将原始文本和目标文本放一起
        all_sents = [src + tgr for src, tgr in self.pairs]
        
        word_counts = Counter()
        # 遍历每一组数据，每一个词，并统计词频并返回counter
        word_counts = count_words(word_counts, all_sents)
        
        vocab = Vocab()
        ###########################################
        #          TODO: module 1 task 2          #
        ###########################################

        # Filter the vocabulary by keeping only the top k tokens in terms of
        # word frequncy in the data set, where k is the maximum vocab size set
        # in "config.py".

        # 遍历topK个词频的词，并添加到vocab
        for word, count in word_counts.most_common(config.max_vocab_size):
            vocab.add_words([word])

        # load预训练的词向量
        if embed_file is not None:
            count = vocab.load_embeddings(embed_file)
            print("%d pre-trained embeddings loaded." % count)

        return vocab


class SampleDataset(Dataset):
    """The class represents a sample set for training.

    """
    def __init__(self, data_pair, vocab):
        self.src_sents = [x[0] for x in data_pair]
        self.trg_sents = [x[1] for x in data_pair]
        self.vocab = vocab
        # Keep track of how many data points.
        self._len = len(data_pair)

    def __getitem__(self, index):
        # 将src转换成id，并对oov做相应处理
        # oov --> oov词在这的顺序按照在x中出现的顺序，id为vocab_size+idx

        x, oov = source2ids(self.src_sents[index], self.vocab)
        return {
            'x': [self.vocab.SOS] + x + [self.vocab.EOS],
            'OOV': oov,
            'len_OOV': len(oov),
            'y': [self.vocab.SOS] +
            abstract2ids(self.trg_sents[index],
                         self.vocab, oov) + [self.vocab.EOS],
            'x_len': len(self.src_sents[index]),
            'y_len': len(self.trg_sents[index])
        }
    def __len__(self):
        return self._len


def collate_fn(batch):
    """Split data set into batches and do padding for each batch.

    Args:
        x_padded (Tensor): Padded source sequences.
        y_padded (Tensor): Padded reference sequences.
        x_len (int): Sequence length of the sources.
        y_len (int): Sequence length of the references.
        OOV (dict): Out-of-vocabulary tokens.
        len_OOV (int): Number of OOV tokens.
    """
    def padding(indice, max_length, pad_idx=0):
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item))
                      for item in indice]
        return torch.tensor(pad_indice)

    data_batch = sort_batch_by_len(batch)

    x = data_batch["x"]
    x_max_length = max([len(t) for t in x])
    y = data_batch["y"]
    y_max_length = max([len(t) for t in y])

    OOV = data_batch["OOV"]
    len_OOV = torch.tensor(data_batch["len_OOV"])

    x_padded = padding(x, x_max_length)
    y_padded = padding(y, y_max_length)

    x_len = torch.tensor(data_batch["x_len"])
    y_len = torch.tensor(data_batch["y_len"])
    return x_padded, y_padded, x_len, y_len, OOV, len_OOV


