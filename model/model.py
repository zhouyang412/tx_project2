#!/usr/bin/env python
# -*- coding: utf-8 -*-



import os
import sys
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))
import config
from utils import timer, replace_oovs

#vocab_size, hidden_size
class Encoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 hidden_size,
                 rnn_drop: float = 0):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embed_size,
                            hidden_size,
                            bidirectional=True,
                            dropout=rnn_drop,
                            batch_first=True)

    def forward(self, x):
        """Define forward propagation for the endoer.

        Args:
            x (Tensor): The input samples as shape (batch_size, seq_len).

        Returns:
            output (Tensor):
                The output of lstm with shape
                (batch_size, seq_len, 2 * hidden_units).
            hidden (tuple):
                The hidden states of lstm (h_n, c_n).
                Each with shape (2, batch_size, hidden_units)
        """
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded)

        return output, hidden


class Attention(nn.Module):
    def __init__(self, hidden_units):
        super(Attention, self).__init__()
        # Define feed-forward layers.
        self.Wh = nn.Linear(2*hidden_units, 2*hidden_units, bias=False)
        self.Ws = nn.Linear(2*hidden_units, 2*hidden_units)

        # wc for coverage feature
        self.wc = nn.Linear(1, 2*hidden_units, bias=False)
        self.v = nn.Linear(2*hidden_units, 1, bias=False)

#     @timer('attention')
    def forward(self,
                decoder_states,
                encoder_output,
                x_padding_masks,
                coverage_vector):
        """Define forward propagation for the attention network.

        Args:
            decoder_states (tuple):
                The hidden states from lstm (h_n, c_n) in the decoder,
                each with shape (1, batch_size, hidden_units)
            encoder_output (Tensor):
                The output from the lstm in the decoder with
                shape (batch_size, seq_len, hidden_units).
            x_padding_masks (Tensor):
                The padding masks for the input sequences
                with shape (batch_size, seq_len).
            coverage_vector (Tensor):
                The coverage vector from last time step.
                with shape (batch_size, seq_len).

        Returns:
            context_vector (Tensor):
                Dot products of attention weights and encoder hidden states.
                The shape is (batch_size, 2*hidden_units).
            attention_weights (Tensor): The shape is (batch_size, seq_length).
            coverage_vector (Tensor): The shape is (batch_size, seq_length).
        """
        # Concatenate h and c to get s_t and expand the dim of s_t.
        h_dec, c_dec = decoder_states
        # (1, batch_size, 2*hidden_units)
        # 对维度变换过后的 hid_state和cell_state进行拼接
        s_t = torch.cat([h_dec, c_dec], dim=2)
        # (batch_size, 1, 2*hidden_units)
        s_t = s_t.transpose(0, 1)

        # 将s_t的维度扩张跟encoder_output一样
        # (batch_size, seq_length, 2*hidden_units)
        s_t = s_t.expand_as(encoder_output).contiguous()

        # calculate attention scores
        # Equation(11).
        # Wh h_* (batch_size, seq_length, 2*hidden_units)
        encoder_features = self.Wh(encoder_output.contiguous())
        # Ws s_t (batch_size, seq_length, 2*hidden_units)
        decoder_features = self.Ws(s_t)
        # (batch_size, seq_length, 2*hidden_units)
        att_inputs = encoder_features + decoder_features

        # Add coverage feature.
        # coverage_vector --> 每一个样本当前时间步之前的encoder中每一个词的att-score之和
        # 加入coverage后，score的计算如原论文的(10)式
        if config.coverage:
            # coverage_features --> (batch_size, seq_length, 1)
            # 若某个词之前一直未被关注，那么其所对应coverage_features将比较小
            # 加入后，鼓励模型可以关注之前未被关注的词，因为某个词一直被关注将可能重复产生高权重
            # 将会影响后续的loss。即该词对应的loss将比较大。
            coverage_features = self.wc(coverage_vector.unsqueeze(2))  # wc c
            # 这里对应公式(11)中的wcci
            att_inputs = att_inputs + coverage_features

        # 这种B-att ，就是拿当前时间步与encoder的output计算att-score所以dim2维度为1
        # (batch_size, seq_length, 1)
        score = self.v(torch.tanh(att_inputs))

        # (batch_size, seq_length)
        attention_weights = F.softmax(score, dim=1).squeeze(2)
        attention_weights = attention_weights * x_padding_masks
        # Normalize attention weights after excluding padded positions.
        # normalization_factor --> 所有权重之和
        normalization_factor = attention_weights.sum(1, keepdim=True)
        # 归一化
        attention_weights = attention_weights / normalization_factor
        # attention_weights.unsqueeze(1) --> (batch_size, 1, seq_length)
        # [batch_size, 1, seq_length] * [batch_size, seq_length, 2*hidden_size] --> [batch_size, 1, 2*hidden_size]
        # context_vector --> (batch_size, 1, 2*hidden_size)
        context_vector = torch.bmm(attention_weights.unsqueeze(1),
                                   encoder_output)
        # (batch_size, 2*hidden_units)
        context_vector = context_vector.squeeze(1)

        # Update coverage vector.
        # 更新coverage，加上当前时间步的权重
        # context_vector --> (batch_size, seq_length)
        if config.coverage:
            coverage_vector = coverage_vector + attention_weights
        # (batch_size, 2*hidden_units), (batch_size, seq_length)
        return context_vector, attention_weights, coverage_vector

class Decoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 hidden_size,
                 enc_hidden_size=None,
                 is_cuda=True):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.DEVICE = torch.device('cuda') if is_cuda else torch.device('cpu')
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

        #
        self.W1 = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.W2 = nn.Linear(self.hidden_size, vocab_size)
        # 这里是为了后续计算Pgen做维度变换
        if config.pointer:
            self.w_gen = nn.Linear(self.hidden_size * 4 + embed_size, 1)
    def forward(self, x_t, decoder_states, context_vector):
        """Define forward propagation for the decoder.

        Args:
            x_t (Tensor):
                The input of the decoder x_t of shape (batch_size, 1).
            decoder_states (tuple):
                The hidden states(h_n, c_n) of the decoder from last time step.
                The shapes are (1, batch_size, hidden_units) for each.
            context_vector (Tensor):
                The context vector from the attention network
                of shape (batch_size,2*hidden_units).

        Returns:
            p_vocab (Tensor):
                The vocabulary distribution of shape (batch_size, vocab_size).
            docoder_states (tuple):
                The lstm states in the decoder.
                The shapes are (1, batch_size, hidden_units) for each.
            p_gen (Tensor):
                The generation probabilities of shape (batch_size, 1).
        """
        # batch_size, 1, embed_size
        decoder_emb = self.embedding(x_t)
        # decoder_output --> [batch_size, 1, hidden_size]
        # decoder_states=(h_n, c_n)
        # h_n, c_n 维度一致--> [1, batch_size, hidden_size]
        # 当前时间步的输出
        decoder_output, decoder_states = self.lstm(decoder_emb, decoder_states)

        # concatenate context vector and decoder state
        # decoder_output --> [batch_size, hidden_size]
        # concat_vector --> [batch_size, 3*hidden_units]
        decoder_output = decoder_output.view(-1, config.hidden_size) # reshape --> [batch_size, hidden_size]
        concat_vector = torch.cat(
            [decoder_output,
             context_vector],
            dim=-1)

        # calculate vocabulary distribution
        # (batch_size, hidden_units)
        # 对输出做维度变换
        FF1_out = self.W1(concat_vector)
        # (batch_size, vocab_size)
        # 隐射到词表大小，输出当前时间步的分布
        FF2_out = self.W2(FF1_out)
        # (batch_size, vocab_size)
        p_vocab = F.softmax(FF2_out, dim=1)

        # Concatenate h and c to get s_t and expand the dim of s_t.
        # h_dec --> hidden_state, c_dec --> cell_state
        # [1, batch_size, hidden_size]
        h_dec, c_dec = decoder_states
        # (1, batch_size, 2*hidden_size)
        s_t = torch.cat([h_dec, c_dec], dim=2)

        p_gen = None
        if config.pointer:
            # Calculate p_gen.
            # 对应原文中的式子(8).
            # 拼接context_vec，当前时间步的hidden_state以及输入
            x_gen = torch.cat([
                context_vector,
                s_t.squeeze(0),
                decoder_emb.squeeze(1)
            ],
                              dim=-1)
            p_gen = torch.sigmoid(self.w_gen(x_gen))
        # p_vocab 为最原始的预测分布
        return p_vocab, decoder_states, p_gen


class ReduceState(nn.Module):
    """
    Since the encoder has a bidirectional LSTM layer while the decoder has a
    unidirectional LSTM layer, we add this module to reduce the hidden states
    output by the encoder (merge two directions) before input the hidden states
    nto the decoder.
    """

    def __init__(self):
        super(ReduceState, self).__init__()

    def forward(self, hidden):
        """The forward propagation of reduce state module.

        Args:
            hidden (tuple):
                Hidden states of encoder,
                each with shape (2, batch_size, hidden_units).

        Returns:
            tuple:
                Reduced hidden states,
                each with shape (1, batch_size, hidden_units).
        """
        # 由于encoder是双向，这里hidden_state和cell_state的第一个维度为2*1=2
        # 这里做一个mean，将第一个维度变成1
        # (2, batch_size, hidden_units)
        h, c = hidden
        h_reduced = torch.sum(h, dim=0, keepdim=True)
        c_reduced = torch.sum(c, dim=0, keepdim=True)

        # (1, batch_size, hidden_units)
        return (h_reduced, c_reduced)


class PGN(nn.Module):
    def __init__(
            self,
            v
    ):
        super(PGN, self).__init__()
        self.v = v
        self.DEVICE = config.DEVICE
        self.attention = Attention(config.hidden_size)
        self.encoder = Encoder(
            len(v),
            config.embed_size,
            config.hidden_size,
        )
        self.decoder = Decoder(len(v),
                               config.embed_size,
                               config.hidden_size,
                               )
        self.reduce_state = ReduceState()

    def load_model(self):

        if (os.path.exists(config.encoder_save_name)):
            print('Loading model: ', config.encoder_save_name)
            self.encoder = torch.load(config.encoder_save_name)
            self.decoder = torch.load(config.decoder_save_name)
            self.attention = torch.load(config.attention_save_name)
            self.reduce_state = torch.load(config.reduce_state_save_name)

        elif config.fine_tune:
            print('Loading model: ', '../saved_model/pgn/encoder.pt')
            self.encoder = torch.load('../saved_model/pgn/encoder.pt')
            self.decoder = torch.load('../saved_model/pgn/decoder.pt')
            self.attention = torch.load('../saved_model/pgn/attention.pt')
            self.reduce_state = torch.load('../saved_model/pgn/reduce_state.pt')

#     @timer('final dist')
    def get_final_distribution(self, x, p_gen, p_vocab, attention_weights,
                               max_oov):
        """Calculate the final distribution for the model.

        Args:
            x: (batch_size, seq_len)
            p_gen: (batch_size, 1)
            p_vocab: (batch_size, vocab_size)
            attention_weights: (batch_size, seq_len)
            max_oov: (Tensor or int): The maximum sequence length in the batch.

        Returns:
            final_distribution (Tensor):
            The final distribution over the extended vocabualary.
            The shape is (batch_size, )
        """

        if not config.pointer:
            return p_vocab

        batch_size = x.size()[0]
        # Clip the probabilities.
        # 对p_gen进行最大最小值截断
        p_gen = torch.clamp(p_gen, 0.001, 0.999)
        # Get the weighted probabilities.
        # 对应原文式子 (9).
        p_vocab_weighted = p_gen * p_vocab
        # (batch_size, seq_len)
        attention_weighted = (1 - p_gen) * attention_weights

        # Get the extended-vocab probability distribution
        # max_oov --> batch中的句子最长的长度
        # extended_size = len(self.v) + max_oovs
        # extension --> [batch_size, max_seq_len]
        extension = torch.zeros((batch_size, max_oov)).float().to(self.DEVICE)
        # (batch_size, extended_vocab_size)
        # p_vocab_extended --> [batch_size, vocab_size + max_seq_len]
        p_vocab_extended = torch.cat([p_vocab_weighted, extension], dim=1)

        # Add the attention weights to the corresponding vocab positions.
        # Refer to equation (9).
        # p_vocab_extended在idx=1的维度上的前vocab个维度为词表内词的预测分布
        # scatter_add_ --> 将src的值，在p_vocab_extended的第一个维度上以x为index，加在相应的位置的值上
        # oov词在x中的id为vocab_size+w.index(x),oov词的权重将会加到extension的位置，当句子的oov词个数没有达到最大
        # oov词数，超过其当前句子的oov词的部分为0，正常的词则会根据x中索引与输出进行相加
        # final_distribution 即包含原词表中的词也包含oov的权重值
        final_distribution = \
            p_vocab_extended.scatter_add_(dim=1,
                                          index=x,
                                          src=attention_weighted)

        return final_distribution

#     @timer('model forward')
    def forward(self, x, x_len, y, len_oovs, batch, num_batches):
        """Define the forward propagation for the seq2seq model.

        Args:
            x (Tensor):
                Input sequences as source with shape (batch_size, seq_len)
            x_len ([int): Sequence length of the current batch.
            y (Tensor):
                Input sequences as reference with shape (bacth_size, y_len)
            len_oovs (Tensor):
                The numbers of out-of-vocabulary words for samples in this batch.
            batch (int): The number of the current batch.
            num_batches(int): Number of batches in the epoch.

        Returns:
            batch_loss (Tensor): The average loss of the current batch.
        """
        # 这里传入的x中若有oov词已经用oov外的id作为其id在utils中有相关处理
        # x_copy中不在词表中的词都用unk填充, 这里的x为encoder输入
        # x --> (batch_size, seq_len)
        x_copy = replace_oovs(x, self.v)
        # x中的pad词部分为0,x中不等于0的部分为True， torch.ne：比较x中的元素，为0的返回1
        x_padding_masks = torch.ne(x, 0).byte().float()

        # encoder embedding
        encoder_output, encoder_states = self.encoder(x_copy)
        # Reduce encoder hidden states.
        # encoder为双向，第一个维度为2，现将其mean成维度1
        # 并初始化为decoder的第一个hidden_state
        # decoder_states --> (1, batch_size, hidden_units)
        decoder_states = self.reduce_state(encoder_states)
        # Initialize coverage vector.
        # coverage_vector --> (batch_size, seq_len)
        coverage_vector = torch.zeros(x.size()).to(self.DEVICE)
        # Calculate loss for every step.
        step_losses = []
        for t in range(y.shape[1]-1):

            # Do teacher forcing.
            # x_t --> (batch_size, 1)
            x_t = y[:, t]
            # 将所有oov词变成unk
            x_t = replace_oovs(x_t, self.v)

            # y_t --> (batch_size, 1)
            y_t = y[:, t+1]
            # Get context vector from the attention network.
            context_vector, attention_weights, coverage_vector = \
                self.attention(decoder_states,
                               encoder_output,
                               x_padding_masks,
                               coverage_vector)
            # Get vocab distribution and hidden states from the decoder.
            # 这里传入decoder的x_t是已经用unkid取代unk词
            # 当预测的时候，上一个时间步选择oov词时，此时间步做unk词输入
            p_vocab, decoder_states, p_gen = self.decoder(x_t.unsqueeze(1),
                                                          decoder_states,
                                                          context_vector)
            # 这里是x而非x_copy，此时包含oov
            final_dist = self.get_final_distribution(x,
                                                     p_gen,
                                                     p_vocab,
                                                     attention_weights,
                                                     torch.max(len_oovs))

            # Get the probabilities predict by the model for target tokens.
            # y_t中的oov词用unk的下标代替
            if not config.pointer:
                y_t = replace_oovs(y_t, self.v)
            # gather --> 在final_dist的第一个维度，按照y_t为idx取出相应的值
            # target_probs --> [batch_size, 1]
            target_probs = torch.gather(final_dist, 1, y_t.unsqueeze(1))
            # 由于这里是每一个时间步取，y_t的shape[1]=1,所以
            # target_probs --> [batch_size, ]
            target_probs = target_probs.squeeze(1)

            # Apply a mask such that pad zeros do not affect the loss
            # mask为y_t中不为pad词的位置为1，否则为0
            mask = torch.ne(y_t, 0).byte()
            # Do smoothing to prevent getting NaN loss because of log(0).
            # loss = - y_true * log p(w) (p(w) = target_probs)
            loss = -torch.log(target_probs + config.eps)

            if config.coverage:
                # Add coverage loss.
                # 对应原文的式子(12),(13)
                # ct_min --> [batch_size, encoder_seq_len]
                # 若某个词经常被关注，其对应的weights将比较大，将产生较大的loss
                # 若词都被雨露均沾，则产生的loss将较小。
                ct_min = torch.min(attention_weights, coverage_vector)
                # cov_loss --> [batch_size,]
                cov_loss = torch.sum(ct_min, dim=1)
                loss = loss + config.LAMBDA * cov_loss

            mask = mask.float()
            # 将pad词对应的loss置0，不计算loss
            loss = loss * mask
            # 将当前时间步的loss放入losses
            step_losses.append(loss)

        # torch.stack(step_losses, 1) --> [batch_size, decoder_seq_len]
        # sample_losses --> [batch_size,]
        # 就是对每一个样本在各个时间步的上loss进行求和
        sample_losses = torch.sum(torch.stack(step_losses, 1), 1)
        # get the non-padded length of each sequence in the batch
        # 将y中的pad词置0，并计算总共的非pad词的数量
        # 每一步loss计算的时候已经将pad的部分置为0，所以这里求平均直接相加求即可
        seq_len_mask = torch.ne(y, 0).byte().float()
        batch_seq_len = torch.sum(seq_len_mask, dim=1)

        # get batch loss by dividing the loss of each batch
        # by the target sequence length and mean
        # 这里的loss已经屏蔽了pad词的计算，故这里的sample_losses为非pad词的总loss
        # 这里求个平均loss
        batch_loss = torch.mean(sample_losses / batch_seq_len)
        return batch_loss