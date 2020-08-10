# -*- coding: utf-8 -*-
"""
@author: Md Mosharaf Hossain
Doctoral Student in CSE at the University of North Texas
"""

import torch
import torch.nn as nn

class model_loss():
    def cross_entropy_loss(self, outputs, labels):
        """
        About: 
            Compute the cross entropy loss.
        Args:
            outputs: (Variable) dimension batch_size*seq_len x num_labels, log softmax output of the model
            labels: (Variable) dimension batch_size x seq_len, where each element is a label in [0, 1, ... num_tag-1], where 0 is id for PAD token.
        
        Returns:
            loss: (Variable) cross entropy loss for all tokens in the batch.
        """   
        
        labels = labels.view(-1)      # view(-1) makes labels a 1-D vector
        loss = nn.CrossEntropyLoss()
        return loss(outputs, labels)  # returns scalar value 
    
    
    def custom_cross_entropy_loss(self, outputs, labels):
        """
        About:
        Compute the cross entropy loss. This function excludes loss terms for PAD tokens.
    
        Args:
            outputs: (Variable) dimension batch_size*seq_len x num_labels - log softmax output of the model
            labels: (Variable) dimension batch_size x seq_len where each element is a label in [0, 1, ... num_tag-1], where 0 is id for PAD token.
    
        Returns:
            loss: (Variable) cross entropy loss for all tokens in the batch
        """
    
        # reshape labels to produce a flat vector of length batch_size*seq_len
        labels = labels.view(-1)
    
        # generate a mask (PAD tokens have label 0) to exclude the loss from those terms
        mask = (labels > 0).float()
        num_tokens = int(torch.sum(mask).item())
    
        # compute cross entropy loss for all tokens (except the PAD tokens), by multiplying with mask.
        return -torch.sum(outputs[range(outputs.shape[0]), labels]*mask)/num_tokens