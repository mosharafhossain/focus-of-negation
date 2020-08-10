# -*- coding: utf-8 -*-
"""
@author: Md Mosharaf Hossain
Doctoral Student in CSE at the University of North Texas
"""
import random
import torch
from allennlp.modules.elmo import batch_to_ids
import numpy as np


def churn_till_maxlen(sent_list, max_len):
    """
    Churn the tokenized sentences up to max_len.
    :param sent_list:
    :param max_len:
    :return:
    """
    batch_tokens = [[word for word in sent[:max_len]] for sent in sent_list]
    batch_sent_lens = [len(sent) for sent in batch_tokens]
    return batch_tokens, batch_sent_lens

def prepare_batch(batch_data, batch_sent_lens):
    """
    Prepare a batch. Max sequence length is max( max_len, the length of the longest sentence in a batch).
    :param batch_data:
    :param batch_sent_lens:
    :return:
    """
    cur_max = max(batch_sent_lens) # get max sequence length for the batch
    batch_data = [sent_tokens[0:cur_max] for sent_tokens, len_ in zip(batch_data, batch_sent_lens)] #list of list
    return batch_data


class batch_preparation():
    def get_a_batch(self, model_obj, data, labels, features_dict, num_steps, batch_size, device, shuffle=False):
        """
        Generate a batch of Data and labels.
        :param model_obj: Contains model parameters
        :param data (Dict): Dictionary of data.
        :param labels (np.ndarray):  Labels with dimension batch_size x seq_len
        :param features_dict (Dict): Features to be considered.
        :param num_steps: Number of steps to complete a single epoch.
        :param batch_size: size of a batch.
        :param device: Apropriate device
        :param shuffle (bool): if the data needs to be shuffled.
        :return:
        """

        data_batch_dict = {}
        data_size = len(labels)             
        order = list(range(data_size)) # generate a list that decides the order in which we go over the data.
        
        if shuffle:
            random.shuffle(order)

        batch_sent_lens = None
        
        for i in range(num_steps):
            indx = 0
            start = i*batch_size
            end   = (i+1)*batch_size if i < num_steps-1 else data_size
            curr_order = order[start:end]

            if "words_org" in features_dict:
                sent_list = list(np.array(data["words_org"])[curr_order])
                batch_tokens, sent_lens= churn_till_maxlen(sent_list, model_obj.max_len)
                character_ids =  batch_to_ids( batch_tokens)
                data_batch_dict[indx] = character_ids
                batch_sent_lens = sent_lens
                indx += 1
            if "sem_role_gold" in features_dict:
                batch_sem_role_gold = data["sem_role_gold"][curr_order]
                batch_sem_role_gold = prepare_batch(batch_sem_role_gold, batch_sent_lens)
                data_batch_dict[indx] = torch.LongTensor(batch_sem_role_gold)
                indx += 1
            if "scope" in features_dict:
                batch_scope = data["scope"][curr_order]
                batch_scope = prepare_batch(batch_scope, batch_sent_lens)
                data_batch_dict[indx] = torch.LongTensor(batch_scope)
                indx += 1                
            if "verb_neg" in features_dict:
                batch_verb_neg = data["verb_neg"][curr_order]
                batch_verb_neg = prepare_batch(batch_verb_neg, batch_sent_lens)
                data_batch_dict[indx] = torch.LongTensor(batch_verb_neg)
                indx += 1

            if "words_bef_only_org" in features_dict:
                sent_list = list(np.array(data["words_bef_only_org"])[curr_order])
                sentence_batch, _= churn_till_maxlen(sent_list, model_obj.max_len)
                character_ids =  batch_to_ids( sentence_batch)
                data_batch_dict[indx] = character_ids
                indx += 1

            if "words_aft_only_org" in features_dict:
                sent_list = list(np.array(data["words_aft_only_org"])[curr_order])
                sentence_batch, _ = churn_till_maxlen(sent_list, model_obj.max_len)
                character_ids = batch_to_ids(sentence_batch)
                data_batch_dict[indx] = character_ids
                indx += 1

            # prepare label batch
            batch_label  = labels[curr_order]
            batch_label  = prepare_batch(batch_label, batch_sent_lens)
            labels_batch = torch.LongTensor(batch_label)

            # shift tensors to appropriate device
            labels_batch = labels_batch.to(device)

            yield data_batch_dict, labels_batch, batch_sent_lens