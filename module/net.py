# -*- coding: utf-8 -*-
"""
@author: Md Mosharaf Hossain
Doctoral Student in CSE at the University of North Texas
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.elmo import Elmo
from torch.autograd import Variable
from module.attention import Attention
from torchcrf import CRF

class nnet(nn.Module):
    def __init__(self, parameters):
        """
        About:
        Initialize parameters for the Neural Network architecture of focus prediction.
        Args:
            parameters: (class instance) contains parameters for the neural network.
        """
        super(nnet, self).__init__()

        self.dtype  = parameters.dtype
        self.device = parameters.device
        self.num_layers = parameters.num_layers
        self.batch_size = parameters.batch_size
        self.max_seq_len = parameters.max_seq_len
        self.num_labels = parameters.num_labels
        self.lstm_units = parameters.lstm_units
        self.dropout = parameters.dropout
        self.num_features = len(parameters.embed_sizes)        
        self.num_directions = 2 if parameters.bidirectional == True else 1
        self.isCRF = parameters.isCRF
        self.isSeq2seq = parameters.isSeq2seq
        self.is_pack_padded = parameters.is_pack_padded
        
        self.is_elmo = parameters.is_elmo
        self.is_bef_aft = parameters.is_bef_aft
        self.options_file = parameters.options_file
        self.weight_file = parameters.weight_file
        
        self.embedding = {}
        embed_size = 0
        for i in range(self.num_features):
            if self.is_elmo and i==0:
                self.embedding[i] = Elmo(self.options_file, self.weight_file, 2, dropout=0).to(self.device)
            else:
                self.embedding[i] = nn.Embedding(parameters.vocab_sizes[i], parameters.embed_sizes[i]).to(self.device, self.dtype)   # the embedding takes as input the vocab_size and the embedding_dim            
            embed_size += parameters.embed_sizes[i]

        self.lstm = nn.LSTM(embed_size, self.lstm_units, dropout=self.dropout, num_layers=self.num_layers, batch_first=True, bidirectional=parameters.bidirectional)       
        linear_unit_features = self.num_directions * self.lstm_units
        self.fc   = nn.Linear(linear_unit_features, self.num_labels)
        if self.isCRF:
            self.crf = CRF(self.num_labels, batch_first=True).to(self.device)

        self.lstm_dropout = nn.Dropout(0.3)

        if self.is_bef_aft:
            self.lstm_units_ctx = parameters.lstm_units_ctx
            self.dropout_ctx    = parameters.dropout_ctx
            self.num_layers_ctx = parameters.num_layers_ctx

            self.seq_len_pre_post = parameters.seq_len_pre_post
            self.post_num_features = len(parameters.post_embed_sizes) 
            self.total_num_features = self.num_features + self.post_num_features
            
            self.post_embedding = Elmo(self.options_file, self.weight_file, 2, dropout=0).to(self.device)
                                                   
            self.post_lstm_bef = nn.LSTM(parameters.post_embed_sizes[0], self.lstm_units_ctx, dropout=self.dropout_ctx, num_layers=self.num_layers_ctx, batch_first=True, bidirectional=True)
            self.post_lstm_aft = nn.LSTM(parameters.post_embed_sizes[1], self.lstm_units_ctx, dropout=self.dropout_ctx, num_layers=self.num_layers_ctx, batch_first=True, bidirectional=True)
            self.word_level_attn_bef = Attention(2 * self.lstm_units_ctx, self.seq_len_pre_post) # multiplied by 2 because of bidirectional lstm
            self.word_level_attn_aft = Attention(2 * self.lstm_units_ctx, self.seq_len_pre_post) # multiplied by 2 because of bidirectional lstm
            linear_unit_features += 2*2*self.lstm_units_ctx # pre_and_post * direction * units
            self.fc   = nn.Linear(linear_unit_features, self.num_labels) 

    def seq_to_seq(self, X):
        seq_len = X.size()[1]

        # make the variable contiguous in memory (a PyTorch artefact).
        X = X.contiguous()     # X dim: batch_size x seq_len x (num_directions*lstm_units)
        X = X.view(-1, X.shape[2])       # X dim: batch_size*seq_len x (num_directions*lstm_units)

        # apply the fully connected layer and obtain representation for each token
        X = self.fc(X) # X dim: batch_size*seq_len x num_labels, X = X.W, (batch_size*seq_len x lstm_units).(lstm_units x num_labels)

        if not self.isCRF:
            # apply log softmax on each token's output (this is recommended over applying softmax since it is numerically more stable)
            X = F.log_softmax(X, dim=1)   # dim: batch_size*seq_len x num_labels
            return X
        else:
            X = X.contiguous()
            X = X.view(self.batch_size, seq_len, self.num_labels) #Dim: batch_size x seq_len x num_labels
            predict = self.crf.decode(X) # Dim (List): batch_size x seq_len
            return predict, X, self.crf

    def get_context_representaion(self, X_dict ):
        indx_bef = self.total_num_features - 2
        indx_aft = self.total_num_features - 1

        # Representation for previous sentence------------------------------------------------------------
        X1 = (X_dict[indx_bef]).to(self.device)
        X1 = self.post_embedding(X1)
        X1 = X1['elmo_representations'][1]  # dim batch_size x batch_seq_len x embed_size
        X1, _ = self.post_lstm_bef(X1) #dim batch_size x batch_seq_len x (direction * lstm units)
        batch, seq_len, lstm_features = X1.size()
        temp = X1.new_zeros((batch, (self.seq_len_pre_post - seq_len), lstm_features))
        X1 = torch.cat([X1, temp], dim=1)
        _, X1 = self.word_level_attn_bef(X1) #dim batch_size  x (direction * lstm units)
        
        # Representation next sentence----------------------------------------------------------------------------------
        X2 = (X_dict[indx_aft]).to(self.device)
        X2 = self.post_embedding(X2)
        X2 = X2['elmo_representations'][1]
        X2, _ = self.post_lstm_aft(X2)  #dim batch_size x batch_seq_len x (direction * lstm units)
        batch, seq_len, lstm_features = X2.size()
        temp = X2.new_zeros((batch, (self.seq_len_pre_post - seq_len), lstm_features))
        X2 = torch.cat([X2, temp], dim=1)
        _, X2 = self.word_level_attn_aft(X2) # dim batch_size  x (direction * lstm units)

        # Prepare contextual representation to be added with the original network
        X = torch.cat([X1, X2],dim=-1)
        X = torch.unsqueeze(X,1)   #dim batch_size  x 1 x  (direction * lstm units)
        batch, _, embed_size = X.size()
        new_data = X.new_zeros((batch, self.seq_len, embed_size)) #dim batch_size x seq_len x (direction * lstm units)
        X = X + new_data     #dim batch_size x seq_len x (direction * lstm units)
        return X
    
    
    def forward(self, X_dict, batch_lens):
        """
        Takes input batch and produce output representations (or most probable sequence for CRF)
        Args:
            X_dict: (Dictionary): can contain batch data of more than one features (e.g tokens, tags of SCOPE).
                    Each feature has dimension batch_size x seq_len.
        Return:
            (Variable): dimension batch_size*seq_len x num_labels with the log probabilities of tokens for each batch
                        If CRF is True: Returns a batch of most probable sequence
        """
        X = (X_dict[0]).to(self.device)
        X = self.embedding[0](X)
        X = X['elmo_representations'][1] # dim batch_size x batch_seq_len x embed_size, batch_seq_len is a variable length maximum sequence length of a batch.

        for i in range(1, self.num_features):
            data = (X_dict[i]).to(self.device) # dim: batch_size x batch_seq_len
            data = self.embedding[i](data) # dim: batch_size x batch_seq_len x embed_sizes[i]
            X = torch.cat([X, data],dim=2) # dim: batch_size x batch_seq_len x sum(embed_sizes), Concate two datasets towards embedding dimension (3rd dimension).
        
        self.batch_size, _, _ = X.size()  # batch_size can be changed in during the validation phase
        hidden = self._init_hidden() #initialize hidden state. if hidden state is initialized to zero, then no operation is required.

        # To make padded items in the sequence not to be shown to the LSTM
        if self.is_pack_padded:
            X = torch.nn.utils.rnn.pack_padded_sequence(X, batch_lens, batch_first=True, enforce_sorted=False)

        # call the BiLSTM network
        X, _ = self.lstm(X, hidden)    # X dim: batch_size x batch_seq_len x (num_directions*lstm_units)

        # undo the packing operation
        if self.is_pack_padded:
            X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True) # X dim: batch_size x batch_seq_len x (num_directions*lstm_units)

        # call context component
        if self.is_bef_aft:
            _, self.seq_len, _ = X.size()
            post_X = self.get_context_representaion(X_dict)
            X = torch.cat([X, post_X],dim=-1)

        X = self.seq_to_seq(X)
        return X
    
    
    def _init_hidden(self):
        # the dimension semantics are [num_directions*num_layers, batch_size, hidden_size]
        init_hidden =  (Variable(torch.rand(self.num_directions*self.num_layers, self.batch_size, self.lstm_units).to(self.device)),
                        Variable(torch.rand(self.num_directions*self.num_layers, self.batch_size, self.lstm_units).to(self.device)))  # init_hidden = (h, c)
        return init_hidden




