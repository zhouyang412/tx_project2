#!/usr/bin/env python
# -*- coding: utf-8 -*-


import random

import torch
import jieba

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))

import config
from model import PGN
from dataset import PairDataset
from utils import source2ids, outputids2words, Beam, timer, add2heap, replace_oovs


class Predict():
    @timer(module='initalize predicter')
    def __init__(self, model, dataset=None, vocab=None, stop_word=None):
        self.DEVICE = config.DEVICE

        if dataset == None:
            dataset = PairDataset(config.data_path,
                                  max_src_len=config.max_src_len,
                                  max_tgt_len=config.max_tgt_len,
                                  truncate_src=config.truncate_src,
                                  truncate_tgt=config.truncate_tgt)

        self.vocab = vocab

        if stop_word == None:
            self.stop_word = []
        else:
            self.stop_word = list(
                set([
                    self.vocab[x.strip()] for x in
                    open(config.stop_word_file
                         ).readlines()
                ]))
        self.model = model.to(config.DEVICE)

    def greedy_search(self,
                      x,
                      max_sum_len,
                      len_oovs,
                      x_padding_masks):
        """Function which returns a summary by always picking
           the highest probability option conditioned on the previous word.

        Args:
            x (Tensor): Input sequence as the source.
            max_sum_len (int): The maximum length a summary can have.
            len_oovs (Tensor): Numbers of out-of-vocabulary tokens.
            x_padding_masks (Tensor):
                The padding masks for the input sequences
                with shape (batch_size, seq_len).

        Returns:
            summary (list): The token list of the result summary.
        """

        # Get encoder output and states.
        # 经过编码器
        encoder_output, encoder_states = self.model.encoder(
            replace_oovs(x, self.vocab))

        # Initialize decoder's hidden states with encoder's hidden states.
        # 编码器第一个维度降维
        decoder_states = self.model.reduce_state(encoder_states)

        # Initialize decoder's input at time step 0 with the SOS token.
        # 初始化第一个输入为SOS
        x_t = torch.ones(1) * self.vocab.SOS
        x_t = x_t.to(self.DEVICE, dtype=torch.int64)
        # summary --> 存储解码结果
        summary = [self.vocab.SOS]

        coverage_vector = torch.zeros((1, x.shape[1])).to(self.DEVICE)
        # Generate hypothesis with maximum decode step.
        # 没有碰到结束符或者小于最长长度则一直解码
        while int(x_t.item()) != (self.vocab.EOS) \
                and len(summary) < max_sum_len:

            context_vector, attention_weights, coverage_vector = \
                self.model.attention(decoder_states,
                                     encoder_output,
                                     x_padding_masks,
                                     coverage_vector)
            p_vocab, decoder_states, p_gen = \
                self.model.decoder(x_t.unsqueeze(1),
                                   decoder_states,
                                   context_vector)
            final_dist = self.model.get_final_distribution(x,
                                                           p_gen,
                                                           p_vocab,
                                                           attention_weights,
                                                           torch.max(len_oovs))
            # Get next token with maximum probability.
            # 每次取概率最高的单词
            x_t = torch.argmax(final_dist, dim=1).to(self.DEVICE)
            decoder_word_idx = x_t.item()
            # 存入summary
            summary.append(decoder_word_idx)
            # 最后替换oov词为unk词
            x_t = replace_oovs(x_t, self.vocab)

        return summary
#     @timer('best k')
    def best_k(self, beam, k, encoder_output, x_padding_masks, x, len_oovs):
        """Get best k tokens to extend the current sequence at the current time step.
        根据beam的上一个词作为decoder的输入，获取在其作为上一个时间步的前提下的当前
        时间步的输出分布。
        Args:
            beam (untils.Beam): The candidate beam to be extended.
            k (int): Beam size.
            encoder_output (Tensor): The lstm output from the encoder.
            x_padding_masks (Tensor):
                The padding masks for the input sequences.
            x (Tensor): Source token ids.
            len_oovs (Tensor): Number of oov tokens in a batch.

        Returns:
            best_k (list(Beam)): The list of best k candidates.

        """
        # use decoder to generate vocab distribution for the next token
        # 取上一个beam中最后一个词作为输入
        x_t = torch.tensor(beam.tokens[-1]).reshape(1, 1)
        x_t = x_t.to(self.DEVICE)

        # Get context vector from attention network.
        # beam.decoder_states --> 上一个时间步的hid_state
        # beam.coverage_vector --> Ct
        context_vector, attention_weights, coverage_vector = \
            self.model.attention(beam.decoder_states,
                                 encoder_output,
                                 x_padding_masks,
                                 beam.coverage_vector)

        # Replace the indexes of OOV words with the index of OOV token
        # to prevent index-out-of-bound error in the decoder.

        p_vocab, decoder_states, p_gen = \
            self.model.decoder(replace_oovs(x_t, self.vocab),
                               beam.decoder_states,
                               context_vector)

        # 获取当前的分布
        final_dist = self.model.get_final_distribution(x,
                                                       p_gen,
                                                       p_vocab,
                                                       attention_weights,
                                                       torch.max(len_oovs))
        # Calculate log probabilities.
        # 对当前的分布值做log
        log_probs = torch.log(final_dist.squeeze())
        # Filter forbidden tokens.
        # 若为第1步，即输入的SOS时
        if len(beam.tokens) == 1:
            # 将这些词的概率置为负无穷大
            forbidden_ids = [
                self.vocab[u"这"],
                self.vocab[u"此"],
                self.vocab[u"采用"],
                self.vocab[u"，"],
                self.vocab[u"。"],
            ]
            log_probs[forbidden_ids] = -float('inf')
        # EOS token penalty. Follow the definition in
        # https://opennmt.net/OpenNMT/translation/beam_search/.
        # x.size()[1] 为输入序列的长度
        # EOS token normalization
        log_probs[self.vocab.EOS] *= \
            config.gamma * x.size()[1] / len(beam.tokens)

        log_probs[self.vocab.UNK] = -float('inf')
        # Get top k tokens and the corresponding logprob.
        # 根据logprobs取出topk个词
        topk_probs, topk_idx = torch.topk(log_probs, k)

        # Extend the current hypo with top k tokens, resulting k new hypos.
        # 将x添加到路径，分数进行相加
        # 包含当前时间步当前词下分数最高的几个路径的beam
        best_k = [beam.extend(x,
                  log_probs[x],
                  decoder_states,
                  coverage_vector) for x in topk_idx.tolist()]

        return best_k

    def beam_search(self,
                    x,
                    max_sum_len,
                    beam_width,
                    len_oovs,
                    x_padding_masks):
        """Using beam search to generate summary.

        Args:
            x (Tensor): Input sequence as the source.
            max_sum_len (int): The maximum length a summary can have.
            beam_width (int): Beam size.
            max_oovs (int): Number of out-of-vocabulary tokens.
            x_padding_masks (Tensor):
                The padding masks for the input sequences.

        Returns:
            result (list(Beam)): The list of best k candidates.
        """
        # run body_sequence input through encoder
        # 编码,初始化Ct，并进行维度变换
        encoder_output, encoder_states = self.model.encoder(
            replace_oovs(x, self.vocab))
        coverage_vector = torch.zeros((1, x.shape[1])).to(self.DEVICE)
        # initialize decoder states with encoder forward states
        decoder_states = self.model.reduce_state(encoder_states)

        # initialize the hypothesis with a class Beam instance.
        # 初始化一个beam对象，token为开始符，encode处理后的hid_state和cell_state作为初始decoder_states
        init_beam = Beam([self.vocab.SOS],
                         [0],
                         decoder_states,
                         coverage_vector)

        # get the beam size and create a list for stroing current candidates
        # and a list for completed hypothesis
        # k为集数大小
        k = beam_width
        curr, completed = [init_beam], []

        # use beam search for max_sum_len (maximum length) steps
        for _ in range(max_sum_len):
            # get k best hypothesis when adding a new token

            # 保存topk个路径的堆
            topk = []
            # 遍历前一个时间步的beams
            for beam in curr:
                # When an EOS token is generated, add the hypo to the completed
                # list and decrease beam size.
                # 当某一个路径已经包含EOS的结束符，则该路径不再进行更新
                # completed包含已经完成的beam对象，包含已经完成的路径
                # 每完成一条路径k-1，剩下的路径每次只需要找topk-n个路径即可
                if beam.tokens[-1] == self.vocab.EOS:
                    completed.append(beam)
                    k -= 1
                    continue
                # 计算由该beam产生的当前时间步的topk的路径
                # 这里将前一个时间步的某一次作为输入，产生的decode_state作为hid_state
                # 送入decoder，产生关于当前时间步的分布输出选取topk个路径
                # 然后插入堆中,堆会根据score来保持一定数量的节点
                for can in self.best_k(beam,
                                       k,
                                       encoder_output,
                                       x_padding_masks,
                                       x,
                                       torch.max(len_oovs)
                                      ):
                    # Using topk as a heap to keep track of top k candidates.
                    # Using the sequence scores of the hypos to campare
                    # and object ids to break ties.
                    # 这里根据seq_score来判断是否插入堆
                    add2heap(topk, (can.seq_score(), id(can), can), k)
            # 根据当前时间步产生的topk个beam对象来更新curr，即当前时间步最好的topk的路径
            # 更新curr，为当前时间步的topk的beam对象。
            # item[2]为当前beam的class对象
            curr = [items[2] for items in topk]
            # stop when there are enough completed hypothesis
            # 若已完成的路径数为k则退出
            if len(completed) == beam_width:
                break
        # When there are not engouh completed hypotheses,
        # take whatever when have in current best k as the final candidates.
        completed += curr
        # sort the hypothesis by normalized probability and choose the best one
        # 根据分数排序
        result = sorted(completed,
                        key=lambda x: x.seq_score(),
                        reverse=True)[0].tokens
        # 返回最佳路径的id
        return result

    @timer(module='doing prediction')
    def predict(self, text, tokenize=True, beam_search=True):
        """Generate summary.

        Args:
            text (str or list): Source.
            tokenize (bool, optional):
                Whether to do tokenize or not. Defaults to True.
            beam_search (bool, optional):
                Whether to use beam search or not.
                Defaults to True (means using greedy search).

        Returns:
            str: The final summary.
        """
        if isinstance(text, str) and tokenize:
            text = list(jieba.cut(text))
#      max_src_len: int = config.max_src_len,
        # x中的oov为对应vocab_size+x.index(w)
        x, oov = source2ids(text, self.vocab)

        if len(x) > max_src_len - 2:
            x = x[: max_src_len - 2]
        x = [self.vocab.SOS] + x + [self.vocab.EOS]
        x = torch.tensor(x).to(self.DEVICE)
        len_oovs = torch.tensor([len(oov)]).to(self.DEVICE)
        x_padding_masks = torch.ne(x, 0).byte().float()
        if beam_search:
            summary = self.beam_search(x.unsqueeze(0),
                                       max_sum_len=config.max_dec_steps,
                                       beam_width=config.beam_size,
                                       len_oovs=len_oovs,
                                       x_padding_masks=x_padding_masks)
        else:
            summary = self.greedy_search(x.unsqueeze(0),
                                         max_sum_len=config.max_dec_steps,
                                         len_oovs=len_oovs,
                                         x_padding_masks=x_padding_masks)
        # 将解码出来的id转换成词语
        summary = outputids2words(summary,
                                  oov,
                                  self.vocab)
        return summary.replace('<SOS>', '').replace('<EOS>', '').strip()


if __name__ == "__main__":
    pred = Predict()
    print('vocab_size: ', len(pred.vocab))
    # Randomly pick a sample in test set to predict.
    with open(config.test_data_path, 'r') as test:
        picked = random.choice(list(test))
        source, ref = picked.strip().split('<sep>')

    print('source: ', source, '\n')
    greedy_prediction = pred.predict(source.split(),  beam_search=False)
    print('greedy: ', greedy_prediction, '\n')
    beam_prediction = pred.predict(source.split(),  beam_search=True)
    print('beam: ', beam_prediction, '\n')
    print('ref: ', ref, '\n')
