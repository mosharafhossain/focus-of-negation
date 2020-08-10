# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 17:58:12 2019
@author: Md Mosharaf Hossain
Doctoral Student in CSE at the University of North Texas
"""

import torch
import module.net as net
import torch.optim as optim
import module.loss as loss

class parameters():
    def __init__(self, cparams, index_dict):                      
        self.learning_rate       = cparams["learning_rate"]
        self.num_layers          = cparams["num_layers"]
        self.bidirectional       = cparams["bidirectional"]
        self.lstm_units          = cparams["lstm_units"]
        self.dropout             = cparams["dropout"]
        self.batch_size          = cparams["batch_size"]
        self.num_epochs          = cparams["num_epochs"]
        self.isCRF               = cparams["isCRF"]
        self.patience            = cparams["patience"]
        self.dtype               = torch.float32 #or torch.float32
        
        self.features_dict       = cparams["features_dict"]
        self.index_dict          = index_dict
        self.max_len             = cparams["max_len"]
        self.isSeq2seq           = cparams["isSeq2seq"]
        self.is_pack_padded      = cparams["is_pack_padded"]
        
        #elmo embeddings
        self.is_elmo             = cparams["is_elmo"]
        self.options_file        = cparams["options_file"]
        self.weight_file         = cparams["weight_file"]
        
        self.best_model_path     = cparams["best_model_path"]

            
        if torch.cuda.is_available()==True: 
            self.device = torch.device("cuda:"+str(cparams["device"]))
        else: 
            self.device = torch.device("cpu")
        
        
        #set vocab sizes and embedding sizes
        indx = 0
        self.vocab_sizes = {}
        self.embed_sizes = {}
        self.embed_matrices = {}
        self.sem_rl_indx = 0
        
        
        # ELMo or GloVe
        if "words_org" in self.features_dict:                   
            self.vocab_sizes[indx]     = len(index_dict["word2index"]) # Number of unique sem_role_gold_grp
            self.embed_sizes[indx]     = self.features_dict["words_org"]  #Set embedding dimension
            self.max_seq_len           = cparams["max_len"] #max_len
            self.is_reduced_seq_len    = False
            indx += 1   
        if "sem_role_gold" in self.features_dict:
            self.vocab_sizes[indx]     = len(index_dict["sem_role_gold2index"]) # Number of unique sem_role_gold_grp
            self.embed_sizes[indx]     = self.features_dict["sem_role_gold"]  #Set embedding dimension
            self.is_reduced_seq_len    = False
            self.sem_rl_indx = indx
            indx += 1              
        if "scope" in self.features_dict:
            self.vocab_sizes[indx]     = len(index_dict["scope2index"]) # Number of unique sem_role_gold_grp
            self.embed_sizes[indx]     = self.features_dict["scope"]   #Set embedding dimension
            self.is_reduced_seq_len    = False            
            indx += 1             
        if "verb_neg" in self.features_dict:
            self.vocab_sizes[indx]     = len(index_dict["verb_neg2index"]) # Number of unique sem_role_gold_grp
            self.embed_sizes[indx]     = self.features_dict["verb_neg"] # 10  #Set embedding dimension
            indx += 1 

        # Initialize parameters for context component
        self.lstm_units_ctx = cparams["lstm_units_ctx"]  # 50
        self.dropout_ctx = cparams["dropout_ctx"]  # 0.3
        self.num_layers_ctx = cparams["num_layers_ctx"]  # 2
        post_indx = 0
        self.post_vocab_sizes = {}
        self.post_embed_sizes = {}
        self.is_bef_aft = False
        
        if "words_bef_only_org" in self.features_dict:
            self.post_vocab_sizes[post_indx]     = len(index_dict["word2index"]) # Number of unique sem_role_gold_grp
            self.post_embed_sizes[post_indx]     = self.features_dict["words_bef_only_org"]  #Set embedding dimension
            self.is_bef_aft                      = True
            self.seq_len_pre_post                = cparams["max_len"]
            post_indx += 1
        
        if "words_aft_only_org" in self.features_dict:
            self.post_vocab_sizes[post_indx]     = len(index_dict["word2index"]) # Number of unique sem_role_gold_grp
            self.post_embed_sizes[post_indx]     = self.features_dict["words_aft_only_org"]  #Set embedding dimension
            self.is_bef_aft                      = True
            self.seq_len_pre_post                = cparams["max_len"]
            post_indx += 1
                                    
            
        self.num_labels = len(index_dict["focus2index"]) #Number of unique focus_grp   
        

class model_objects():
    def __init__(self, params, best_model_path):
        self.model = net.nnet(params).to(params.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=params.learning_rate, weight_decay=0)
        self.loss_fn = loss.model_loss().custom_cross_entropy_loss  #customs loss function
        self.isCRF = params.isCRF  
        self.isSeq2seq = params.isSeq2seq
        self.device = params.device
        self.sem_rl_indx = params.sem_rl_indx
        self.patience = params.patience
        self.best_model_path = params.best_model_path        
        self.index_dict = params.index_dict
        self.is_elmo = params.is_elmo
        self.max_len = params.max_len
        
