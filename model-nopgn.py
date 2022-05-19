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

        ###########################################
        #          TODO: module 2 task 2.1        #
        ###########################################
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

        ###########################################
        #          TODO: module 2 task 2.2        #
        ###########################################
        # Add coverage feature.
        if config.coverage:


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
        # [1, seq_length] * [seq_length, 2*hidden_size] --> [1, 2*hidden_size]
        # (batch_size, 1, 2*hidden_size)
        context_vector = torch.bmm(attention_weights.unsqueeze(1),
                                   encoder_output)
        # (batch_size, 2*hidden_units)
        context_vector = context_vector.squeeze(1)

        ###########################################
        #          TODO: module 2 task 2.3        #
        ###########################################
        # Update coverage vector.
        if config.coverage:

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

        self.W1 = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.W2 = nn.Linear(self.hidden_size, vocab_size)

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
        # h_n,c_n --> [1, batch_size, hidden_size]
        decoder_output, decoder_states = self.lstm(decoder_emb, decoder_states)

        # concatenate context vector and decoder state
        # decoder_output --> [batch_size, hidden_size]
        # concat_vector --> [batch_size, 3*hidden_units]
        decoder_output = decoder_output.view(-1, config.hidden_size)
        concat_vector = torch.cat(
            [decoder_output,
             context_vector],
            dim=-1)

        # calculate vocabulary distribution
        # (batch_size, hidden_units)
        FF1_out = self.W1(concat_vector)
        # (batch_size, vocab_size)
        FF2_out = self.W2(FF1_out)
        # (batch_size, vocab_size)
        p_vocab = F.softmax(FF2_out, dim=1)

        # Concatenate h and c to get s_t and expand the dim of s_t.
        # h_dec --> hidden_state, c_dec --> cell_state
        # [1, batch_size, hidden_size]
        h_dec, c_dec = decoder_states
        # (1, batch_size, 2*hidden_size)
        s_t = torch.cat([h_dec, c_dec], dim=2)

        ###########################################
        #          TODO: module 2 task 1.2        #
        ###########################################
        p_gen = None
        if config.pointer:
            # Calculate p_gen.
            # Refer to equation (8).


        return p_vocab, decoder_states, p_gen


class ReduceState(nn.Module):
    """
    Since the encoder has a bidirectional LSTM layer while the decoder has a
    unidirectional LSTM layer, we add this module to reduce the hidden states
    output by the encoder (merge two directions) before input the hidden states
    nto the decoder.
    """
    ###########################################
    #          TODO: module 2 task 5          #
    ###########################################

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


class Seq2seq(nn.Module):
    def __init__(
            self,
            v
    ):
        super(Seq2seq, self).__init__()
        self.v = v
        self.DEVICE = torch.device("cuda" if config.is_cuda else "cpu")
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
        self.lambda_cov = torch.tensor(1.,
                                       requires_grad=False,
                                       device=self.DEVICE)

    def load_model(self):
        """Load saved model if there exits one.
        """        
        if (os.path.exists(config.encoder_save_name)):
            self.encoder = torch.load(config.encoder_save_name)
            self.decoder = torch.load(config.decoder_save_name)
            self.attention = torch.load(config.attention_save_name)
            self.reduce_state = torch.load(config.reduce_state_save_name)

    def forward(self, x, x_len, y, len_oovs, batch):
        """Define the forward propagation for the seq2seq model.

        Args:
            x (Tensor):
                Input sequences as source with shape (batch_size, seq_len)
            x_len ([int): Sequence length of the current batch.
            y (Tensor):
                Input sequences as reference with shape (bacth_size, y_len)
            len_oovs (int):
                The number of out-of-vocabulary words in this sample.
            batch (int): The number of the current batch.

        Returns:
            batch_loss (Tensor): The average loss of the current batch.
        """

        ###########################################
        #          TODO: module 2 task 4          #
        ###########################################
        # 生成一个跟x形状一样的矩阵用unk的id来填充
        oov_token = torch.full(x.shape, self.v.UNK).long().to(self.device)

        # 若x中的idx是大于词表的即在词表之外的，统一用unk的idx，否则就是其对应的idx
        x_copy = torch.where(x > len(self.v) - 1, oov_token, x)

        # torch.ne为对应的值不相等时返回true
        # 这里不是pad的位置都返回True
        x_padding_masks = torch.ne(x_copy, 0).byte().float()

        encoder_output, encoder_states = self.encoder(x_copy)
        # Reduce encoder hidden states.
        # 这里对第一个维度进行压缩为1
        decoder_states =  self.reduce_state(encoder_states)

        # Calculate loss for every step.
        step_losses = []
        # 最后一个词之后停止所以这里减1
        # y-->(bacth_size, y_len)
        for t in range(y.shape[1]-1):
            # 这里拿t去预测t+1
            # [batch_size, ]
            decoder_input_t = y[:, t] # x_t
            decoder_target_t = y[:, t+1]  # y_t
            # Get context vector from the attention network.
            # (batch_size, 2*hidden_units), (batch_size, seq_length)
            context_vector, attention_weights = self.attention(
                decoder_states, encoder_output, x_padding_masks
            )
            # Get vocab distribution and hidden states from the decoder.
            # (batch_size, vocab_size), (1, batch_size, hidden_size)
            p_vocab, decoder_states = self.decoder(
                decoder_input_t.unsqueeze(1), decoder_states, encoder_output
            )

            # Get the probabilities predict by the model for target tokens.
            # torch.gather(input, dim, index):在input的dim维度上按index取相应的值
            # [batch_size, 1]
            target_probs = torch.gather(p_vocab,
                                        1,
                                        decoder_target_t.unsqueeze(1))
            # [batch_size,1 , 1]
            target_probs = target_probs.squeeze(1)
            # Apply a mask such that pad zeros do not affect the loss
            # 目标词中为pad词的置0
            mask = torch.ne(decoder_target_t, 0).byte()
            # Do smoothing to prevent getting NaN loss because of log(0).
            # -y * log(p)
            loss = -torch.log(target_probs + config.eps)
            mask = mask.float()
            # batch中此时间步为pad词的不产生loss
            loss = loss * mask
            step_losses.append(loss)

        # torch.stack(step_losses, 1) --> [batch_size, decoder_seq_len]
        # sample_losses --> [batch_size,]
        # 就是对每一个样本在各个时间步的上loss进行求和
        sample_losses = torch.sum(torch.stack(step_losses, 1), 1)
        # get the non-padded length of each sequence in the batch
        # 将目标词矩阵中pad词的部分置0
        seq_len_mask = torch.ne(y, 0).byte().float()
        # 所有非pad词的综合
        batch_seq_len = torch.sum(seq_len_mask, dim=1)

        # get batch loss by dividing the loss of each batch
        # by the target sequence length and mean
        # loss为总的loss除以所有非pad词数
        batch_loss = torch.mean(sample_losses / batch_seq_len)
        return batch_loss
