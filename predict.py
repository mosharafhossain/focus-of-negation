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

# Extrac vocabs
with open(cparams["vocab_loc"], "rb") as file_obj:
    vocab_dict = pickle.load(file_obj)  
    token_dict = vocab_dict["token_dict"]
    index_dict = vocab_dict["index_dict"]


# Set parameters___________________________________________________________________________
index2token = index_dict["index2focus"]
params = param.parameters(cparams, index_dict)
model_obj = param.model_objects(params, cparams["best_model_path"])


# Evaluation_______________________________________________________________________________________________
eval_obj = evaluate.evaluate_model()
print("Loading best model...")
model_obj.model.load_state_dict(torch.load(model_obj.best_model_path, map_location=model_obj.device))  #loading the best model, follow this: https://discuss.pytorch.org/t/save-and-load-model/6206/17


# Test data preparation______________________________________________________________________
prep_obj = data_prep_focus.data_for_training()
test_proc_data, test_y, te_sent_length, test_sem_role_gold_grp_cnt =  prep_obj.get_data_for_validation(cparams["test_file"], cparams["max_len"], index_dict, token_dict, srl_dict, cparams["isLower"], cparams["label_name"])
test_proc_data,_ = prep_obj.tag_sematic_roles(test_proc_data, index_dict["sem_role_gold_v22index"], "sem_role_gold_v2")
#integrate scope info in test data
with open(cparams["scope_prediction"], "rb") as file_obj:
    pred_scope_dict = pickle.load(file_obj)  #keys: tr_pred_scope,te_pred_scope
test_proc_data, _, _ = prep_obj.integrate_scope_data(test_proc_data, pred_scope_dict["te_pred_scope"], "test", token_dict, index_dict, cparams["max_len"], test_sem_role_gold_grp_cnt)
test_y = np.argmax(test_y, axis=2)
#---------------------------------------------------------------------------------------------------


# on Test data_____________________
test_data_iterator = batch.batch_preparation().get_a_batch(model_obj, test_proc_data, test_y, cparams["features_dict"], 1, len(test_y), params.device) #num_steps = 1 for full data, batch_size = len(test_y)
test_x, test_y, sent_lens = next(test_data_iterator)

# Store predicted outcome in files
rows, cols = test_x[model_obj.sem_rl_indx].shape
test_pred = eval_obj.predict_test_single(model_obj, test_x, index2token, rows, cols, sent_lens)
test_file_obj = open(cparams["test_file"], "r")   
dp_obj = data_prep_focus.data_preparation() 
test_data_objs = dp_obj.data_load(test_file_obj)
sent_size = len(test_proc_data["words"])*3
new_obj_list = dp_obj.create_new_obj_list(test_data_objs, test_pred, cparams["max_len"], sent_size)
dp_obj.print_to_file(new_obj_list, cparams["output_file_path"]) 
	
