# -*- coding: utf-8 -*-
"""
@author: Md Mosharaf Hossain
Doctoral Student in CSE at the University of North Texas
"""
import module.batch as batch
import module.train as training
import module.evaluate as evaluate
import module.parameters as param
import pickle
import torch
import numpy as np
import data_prep_focus
import random
import os
import argparse
import json 


def set_seed(seed=42):
	os.environ['PYTHONHASHSEED'] = str(seed)
	random.seed(seed)	
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # required for using multi-GPUs.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


# Command line arguments
argParser = argparse.ArgumentParser()
argParser.add_argument("-c", "--config_path", help="path of the configuration file", required=True)      
args        = argParser.parse_args()
config_path = args.config_path


# Read parameters from json file
with open(config_path) as json_file_obj: 
	cparams = json.load(json_file_obj)


# Set the seed    
set_seed(cparams["seed"]) 


# Load gold semantic roles (Extracted from propbank dataset)
with open(cparams["gold_srl_path"], "rb") as file_obj:
	srl_dict = pickle.load(file_obj)


# Training data preparation__________________________________________________________________
tr_obj = data_prep_focus.data_for_training()
tr_proc_data, train_y_orig, token_dict, index_dict, tr_sent_length, tr_sem_role_gold_grp_cnt =  tr_obj.get_data_for_training(cparams["train_dev_file"], cparams["max_len"], srl_dict, cparams["isLower"], cparams["label_name"])
tr_proc_data, _ = tr_obj.tag_sematic_roles(tr_proc_data, index_dict["sem_role_gold_v22index"], "sem_role_gold_v2")


# retrieving the scope prediction
with open(cparams["scope_prediction"], "rb") as file_obj:
	pred_scope_dict = pickle.load(file_obj)
tr_proc_data, token_dict, index_dict = tr_obj.integrate_scope_data(tr_proc_data, pred_scope_dict["tr_pred_scope"], "training", token_dict, index_dict, cparams["max_len"], tr_sem_role_gold_grp_cnt)
train_y = np.argmax(train_y_orig, axis=2)
tr_proc_data, train_y, val_proc_data, val_y = tr_obj.train_dev_split(tr_proc_data, train_y, cparams["val_percent"], cparams["isSeq2seq"]) #write this function
vocab_dict = {"token_dict":token_dict, "index_dict":index_dict}


# Save vocab
with open(cparams["vocab_loc"], "wb") as file_obj:
	pickle.dump(vocab_dict, file_obj)


# Set parameters___________________________________________________________________________
index2token = index_dict["index2focus"]
params = param.parameters(cparams, index_dict)
model_obj = param.model_objects(params, cparams["best_model_path"])


# Setting parameter for data evaluation
val_data_iterator = batch.batch_preparation().get_a_batch(model_obj, val_proc_data, val_y, cparams["features_dict"], 1, len(val_y), params.device) #num_steps = 1 for full data, batch_size = len(val_y)
val_x, val_y, sent_lens = next(val_data_iterator)
val_data_eval = {"val_x": val_x, "val_y": val_y, "sent_lens":sent_lens, "index2token": index2token, "label_tag": cparams["label_tag"], "pad_name": cparams["pad_name"]}
data_eval = {"val": val_data_eval}


# Model training and saving the best model
training.model_train().model_train(params, model_obj, tr_proc_data, train_y, cparams["features_dict"], data_eval)