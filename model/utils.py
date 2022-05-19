#!/usr/bin/env python
# -*- coding: utf-8 -*-


import time
import heapq

import torch

import config
import numpy as np


def timer(module):
    """Decorator function for a timer.

    Args:
        module (str): Description of the function being timed.
    """
    def wrapper(func):
        """Wrapper of the timer function.

        Args:
            func (function): The function to be timed.
        """
        def cal_time(*args, **kwargs):
            """The timer function.

            Returns:
                res (any): The returned value of the function being timed.
            """
            t1 = time.time()
            res = func(*args, **kwargs)
            t2 = time.time()
            cost_time = t2 - t1
            print(f'{cost_time} secs used for ', module)
            return res
        return cal_time
    return wrapper


def simple_tokenizer(text):
    return text.split()


def count_words(counter, text):
    '''Count the number of occurrences of each word in a set of text'''
    # 根据counter的数据类型，分别遍历text中的词，并统计词频
    if type(counter) == type(dict()):
        for sentence in text:
            for word in sentence:
                counter[word] += 1
    else:
        for sentence in text:
            for word in sentence:
                counter.update(word)  
                
    return counter


def sort_batch_by_len(data_batch):
    """

    Args:
        data_batch (Tensor): Batch before sorted.

    Returns:
        Tensor: Batch after sorted.
    """
    res = {'x': [],
           'y': [],
           'x_len': [],
           'y_len': [],
           'OOV': [],
           'len_OOV': []}
    for i in range(len(data_batch)):
        res['x'].append(data_batch[i]['x'])
        res['y'].append(data_batch[i]['y'])
        res['x_len'].append(len(data_batch[i]['x']))
        res['y_len'].append(len(data_batch[i]['y']))
        res['OOV'].append(data_batch[i]['OOV'])
        res['len_OOV'].append(data_batch[i]['len_OOV'])

    # Sort indices of data in batch by lengths.
    sorted_indices = np.array(res['x_len']).argsort()[::-1].tolist()

    data_batch = {
        name: [_tensor[i] for i in sorted_indices]
        for name, _tensor in res.items()
    }
    return data_batch


def outputids2words(id_list, source_oovs, vocab):
    """
        Maps output ids to words, including mapping in-source OOVs from
        their temporary ids to the original OOV string (applicable in
        pointer-generator mode).
        Args:
            id_list: list of ids (integers)
            vocab: Vocabulary object
            source_oovs:
                list of OOV words (strings) in the order corresponding to
                their temporary source OOV ids (that have been assigned in
                pointer-generator mode), or None (in baseline mode)
        Returns:
            words: list of words (strings)
    """

    ###########################################
    #          TODO: module 1 task 4          #
    ###########################################

    # source_oovs为oov的词，id_list中为对应的id
    words = []

    for i in id_list:
        try:
            # 尝试在词表中找词，找不到则为oov
            w = vocab.index2word[i]
        except IndexError:
            assert_msg = "ERROR ID cant find"
            assert source_oovs is not None, assert_msg
            # 在source2ids中，oov词对应的在ids中的id为vocab_size+x.index(w)
            # 所以通过该相减操作，找到oov词在src中的下标
            # 并根据下标从src找到相应的w
            source_oov_idx = i - vocab.size()
            try:
                w = source_oovs[source_oov_idx]
            except ValueError:  # i doesn't correspond to an source oov
                raise ValueError(
                    'Error: model produced word ID %i corresponding to source OOV %i \
                     but this example only has %i source OOVs'
                    % (i, source_oov_idx, len(source_oovs)))

        words.append(w)

    return ' '.join(words)


def source2ids(source_words, vocab):
    """Map the source words to their ids and return a list of OOVs in the source.
    Args:
        source_words: list of words (strings)
        vocab: Vocabulary object
    Returns:
        ids:
        A list of word ids (integers); OOVs are represented by their temporary
        source OOV number. If the vocabulary size is 50k and the source has 3
        OOVs tokens, then these temporary OOV numbers will be 50000, 50001,
        50002.
    oovs:
        A list of the OOV words in the source (strings), in the order
        corresponding to their temporary source OOV numbers.
    """

    ###########################################
    #          TODO: module 1 task 3          #
    ###########################################
    # 这是针对src的转换id处理
    # 这里oov的idx是相对当前样本x的
    # oov中包含该src中的oov的具体的词
    # oov在id中为oov词在src中的下标加上词表大小
    ids = []
    oovs = []
    unk_id = vocab["<UNK>"]

    for word in source_words:
        i = vocab[word]
        if i == unk_id:
            # 不在oov列表中就添加到列表，否则直接取对应下标
            if word not in oovs:
                oovs.append(word)
            # oov_num 为当前的oov词的个数 - 1
            oov_num = oovs.index(word)
            # 对应oov词id为当前词表大小+当前列表的下标
            ids.append(vocab.size() + oov_num)
        else:
            # 若不是oov则直接返回对应的idx
            ids.append(i)
 
    return ids, oovs

def abstract2ids(abstract_words, vocab, source_oovs):
    """Map tokens in the abstract (reference) to ids.
       OOV tokens in the source will be remained.

    Args:
        abstract_words (list): Tokens in the reference.
        vocab (vocab.Vocab): The vocabulary.
        source_oovs (list): OOV tokens in the source.

    Returns:
        list: The reference with tokens mapped into ids.
    """
    # 这个函数用来处理ref（参考输出）转换成id并处理oov

    ids = []
    unk_id = vocab.UNK
    # 遍历参考输出，如果出现oov则看其是否出现在输入中
    for w in abstract_words:
        i = vocab[w]
        if i == unk_id:  # If w is an OOV word
            # 如果发现参考输出中出现的词也在输入中则给予其一个临时的id
            # 该id的值为vocab_size + 词在src中得出的oov列表中的下标
            if w in source_oovs:  # If w is an in-source OOV
                # Map to its temporary source OOV number
                vocab_idx = vocab.size() + source_oovs.index(w)
                ids.append(vocab_idx)
            # 如果改词不在输入中则将其置为unk
            else:  # If w is an out-of-source OOV
                ids.append(unk_id)  # Map to the UNK token id
        else:
            ids.append(i)
    return ids

class Beam(object):
    def __init__(self,
                 tokens,
                 log_probs,
                 decoder_states,
                 coverage_vector):
        self.tokens = tokens
        self.log_probs = log_probs
        self.decoder_states = decoder_states
        self.coverage_vector = coverage_vector

    def extend(self,
               token,
               log_prob,
               decoder_states,
               coverage_vector):
        return Beam(tokens=self.tokens + [token],
                    log_probs=self.log_probs + [log_prob],
                    decoder_states=decoder_states,
                    coverage_vector=coverage_vector)

    def seq_score(self):
        """
        This function calculate the score of the current sequence.
        The scores are calculated according to the definitions in
        https://opennmt.net/OpenNMT/translation/beam_search/.
        1. Lenth normalization is used to normalize the cumulative score
        of a whole sequence.
        2. Coverage normalization is used to favor the sequences that fully
        cover the information in the source. (In this case, it serves different
        purpose from the coverage mechanism defined in PGN.)
        3. Alpha and beta are hyperparameters that used to control the
        strengths of ln and cn.
        """
        len_Y = len(self.tokens)
        # Lenth normalization
        ln = (5+len_Y)**config.alpha / (5+1)**config.alpha
        cn = config.beta * torch.sum(  # Coverage normalization
            torch.log(
                config.eps +
                torch.where(
                    self.coverage_vector < 1.0,
                    self.coverage_vector,
                    torch.ones((1, self.coverage_vector.shape[1])).to(torch.device(config.DEVICE))
                )
            )
        )

        score = sum(self.log_probs) / ln + cn
        return score

    def __lt__(self, other):
        return self.seq_score() < other.seq_score()

    def __le__(self, other):
        return self.seq_score() <= other.seq_score()



def add2heap(heap, item, k):
    """Maintain a heap with k nodes and the smallest one as root.

    Args:
        heap (list): The list to heapify.
        item (tuple):
            The tuple as item to store.
            Comparsion will be made according to values in the first position.
            If there is a tie, values in the second position will be compared,
            and so on.
        k (int): The capacity of the heap.
    """
    # 最小值作为root节点
    # 堆中保持只有K个节点，当数量大于K则弹出最小值
    # 这里的最小值根据第一个值也就是seq_score

    if len(heap) < k:
        heapq.heappush(heap, item)
    else:
        heapq.heappushpop(heap, item)

def replace_oovs(in_tensor, vocab):
    """Replace oov tokens in a tensor with the <UNK> token.

    Args:
        in_tensor (Tensor): The tensor before replacement.
        vocab (vocab.Vocab): The vocabulary.

    Returns:
        Tensor: The tensor after replacement.
    """
    # oov_token --> shape与in_tensor相同，用unk的id填充
    oov_token = torch.full(in_tensor.shape, vocab.UNK, dtype=torch.long).to(config.DEVICE)
    # idx大于词表大小的部分，用unk填充，为OOV词
    out_tensor = torch.where(in_tensor > len(vocab) - 1, oov_token, in_tensor)
    return out_tensor