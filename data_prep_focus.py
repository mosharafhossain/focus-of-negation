# -*- coding: utf-8 -*-
"""
@author: Md Mosharaf Hossain
Doctoral Student in CSE at the University of North Texas
"""

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from collections import defaultdict
from sklearn.model_selection import train_test_split
import copy
import spacy
import os
nlp = spacy.load("en_core_web_sm")

PAD = "PAD"
UNKNOWN = "UNK"

class data_structure:
    def __init__(self, line):
        tokens = line.split()
        size = len(tokens)
        self.col0 = tokens[0]
        self.col1 = tokens[1]
        self.col2 = tokens[2]
        self.col3 = tokens[3]
        self.word = tokens[4]
        self.word_num = tokens[5]
        self.pos = tokens[6]
        self.name_entity = tokens[7]
        self.chunk = tokens[8]
        self.parse_tree = tokens[9]
        self.dep_pnum = tokens[10] 
        self.dep_label = tokens[11]
        self.sem_role = tokens[12:size-2]
        self.verbal_neg = tokens[size-2]
        self.focus = tokens[size-1]
        
        
class data_preparation:
    def data_load(self, file):
        obj_list = []
        for line in file:
            if len(line) >= 14:
                obj = data_structure(line)
                obj_list.append(obj)
        file.close()                
        return obj_list

    def get_syntactical_info(self, sentence):  
        sentence = " ".join(sentence) #creating a sentence from a list of words
        sentence = nlp(sentence)
        upos = [token.pos_ for token in sentence]
        synt_dep  = [token.dep_ for token in sentence]
        syn_child = [len([child for child in token.children]) for token in sentence]
        return upos, synt_dep, syn_child
    
    def get_data_details(self, obj_list):
        data_dict = {}
        data_dict_info = {}
        
        word_list = []
        word_num_list = []
        pos_list = []
        name_entity_list = []
        chunk_list = []
        parse_tree_list =[]
        dep_pnum_list = []
        dep_label_list = []
        sem_role_dict = defaultdict(list)
        verbal_neg_list = []
        focus_list = []
        sent_number = 0
        num_focus = 0
        isVN = False
        VN_col = -1
        
        for i in range(len(obj_list)):
                        
            # Create a unique tuple for dictionary
            info_tuple = (obj_list[i].col2, obj_list[i].col3)   # col2 is section number(02-24, propbank), col3 sentence number
            
            # Extract features information
            word_list.append( obj_list[i].word.strip() )
            word_num_list.append( int(obj_list[i].word_num.strip()) )
            pos_list.append( obj_list[i].pos.strip() )            
            name_entity_list.append( obj_list[i].name_entity.strip() )
            chunk_list.append( obj_list[i].chunk.strip() )
            parse_tree_list.append( obj_list[i].parse_tree.strip() )
            dep_pnum_list.append( obj_list[i].dep_pnum.strip() )
            dep_label_list.append( obj_list[i].dep_label.strip() )
            
            if obj_list[i].verbal_neg.strip() == "N":
                verbal_neg_list.append( "I_N" ) #Verbal Neg
                isVN = True
            else:
                verbal_neg_list.append( "O_N" ) #Not Verbal Neg
            
            if obj_list[i].focus.strip() == "FOCUS":
                focus_list.append( "I_F" ) #Inside focus
                num_focus += 1
            else:
                focus_list.append( "O_F" ) # Not a focus
            
            sem_role = obj_list[i].sem_role
            for j in range(len(sem_role) ):                
                sem_role_dict[j].append( sem_role[j].strip() )
                if isVN == True and sem_role[j].strip() == "(V*)" and VN_col == -1: #picking first VN for first (V*). (V*) can be two words, eg. "set up", "put up"
                    VN_col = j  #Verval neg column
                    isVN = False
                
            if (i == len(obj_list)-1) or ( i+1 < len(obj_list) and int(obj_list[i+1].word_num.strip()) == 1): 
                
                num_words = int(obj_list[i].word_num.strip()) 
                upos_list, _, _ = self.get_syntactical_info(word_list)
                
                data_dict[sent_number] = [num_words, VN_col, num_focus, word_list, word_num_list, pos_list, upos_list, name_entity_list, chunk_list, parse_tree_list, dep_pnum_list, dep_label_list, sem_role_dict, verbal_neg_list, focus_list]
                data_dict_info[sent_number] = info_tuple
                
                word_list = []
                word_num_list = []
                pos_list = []
                name_entity_list = []
                chunk_list = []
                parse_tree_list =[]
                dep_pnum_list = []
                dep_label_list = []
                sem_role_dict = defaultdict(list)
                verbal_neg_list = []
                focus_list = []
                sent_number += 1
                num_focus = 0
                isVN = False
                VN_col = -1
                
        return data_dict, data_dict_info
    
    def remove_C_R(self, token):
        if ("C-" in token) or ("R-" in token):
            pos = token.find("-")+1
            return "(" + token[pos:]
        else: 
            return token

    def get_sem_role_v2(self, sem_role):
        l = len(sem_role)
        sem_role_v2 = ["*"]*l
        isMulti = False
        
        for i in range(l):
            cur_sr = sem_role[i].strip()
            if cur_sr[0] == "(" and cur_sr[-1] == ")":
                sem_role_v2[i] = self.remove_C_R(cur_sr)
            if cur_sr[0] == "(" and cur_sr[-1] == "*":
                saved_sr = cur_sr + ")"
                isMulti = True
            if isMulti == True:
                assert (saved_sr != "")
                sem_role_v2[i] = self.remove_C_R(saved_sr)
            if cur_sr == "*)":
                isMulti = False
                saved_sr = ""
        return sem_role_v2
                
    
    def get_sem_role_v3(self, sem_role):
        l = len(sem_role)
        sem_role_v3 = ["O_R"]*l
        i = start = 0
        j = end = l-1
        isStart = isEnd = False
        while i < l or j >=0 :
            if isStart == False and sem_role[i].strip() != "*": 
                start = i
                isStart = True
            else: i += 1
            
            if isEnd == False and sem_role[j].strip() != "*": 
                end = j
                isEnd = True
            else: j -= 1
            
            if isStart == True and isEnd == True:
                break
            
        for i in range(l):
            if i >= start and i <= end:
                sem_role_v3[i] = "I_R"
            
        return sem_role_v3
    
    def get_sem_role_grp(self, row, gold_srl_v2, focus, verb_neg):
        gold_srl_grp = []
        gold_srl_grp_cnt = []
        focus_grp = []
        verb_neg_grp = []
        l = len(gold_srl_v2)
        
        curr_sem_role = gold_srl_v2[0]
        curr_focus = focus[0]
        curr_verb_ng = verb_neg[0]
        counter = 1
        er_cnt = 0

        for i in range(1, l):
            if curr_sem_role != gold_srl_v2[i]:
                gold_srl_grp.append(curr_sem_role)
                gold_srl_grp_cnt.append(counter)
                focus_grp.append(curr_focus)
                verb_neg_grp.append(curr_verb_ng)
                
                curr_sem_role = gold_srl_v2[i]
                curr_focus = focus[i]
                curr_verb_ng = verb_neg[i]
                counter = 1
            elif curr_sem_role == gold_srl_v2[i]:
                counter += 1
                if curr_focus != focus[i]: 
                    er_cnt += 1            
                
        gold_srl_grp.append(curr_sem_role)
        gold_srl_grp_cnt.append(counter)
        focus_grp.append(curr_focus)
        verb_neg_grp.append(curr_verb_ng)
        assert len(gold_srl_grp)==len(gold_srl_grp_cnt)==len(focus_grp) == len(verb_neg_grp)
        
        return gold_srl_grp, gold_srl_grp_cnt, focus_grp, verb_neg_grp, er_cnt
    
    
    def text_match(self, sent1, sent2):
       result = []
       for i in range(len(sent1)):
           if sent1[i] in sent2:
               result.append(sent1[i])
       return result


    def data_for_focus_resolution(self, data_dict, data_dict_info, srl_dict):
        data = defaultdict(list)
        sent_length = []
        a = 1; pos = 1; d = 3
        #counter = 0
        for i in range(len(data_dict) ):
            num_words    = data_dict[i][0]   # index 0 stores size of the sentences            
            vn_index     = data_dict[i][1]   # index 1 stores verval negation column of a sentence
            #num_focus    = data_dict[i][2]   # index 2 stores number of focus of a sentence    
            
            if i == (a+(pos-1)*d):  #get the position of current sentence by the formula a+(n-1)d
                data["words"].append(data_dict[i][3]) # index 3 stores words list of a sentence
                data["words_bef"].append(data_dict[i-1][3] + data_dict[i][3]) # Adding previous sentence with current sentence
                data["words_aft"].append(data_dict[i][3] + data_dict[i+1][3]) # Adding next sentence with current sentence
                data["words_bef_aft"].append(data_dict[i-1][3] + data_dict[i][3] + data_dict[i+1][3]) # Adding previous and next sentence with current sentence
                data["words_bef_only"].append(data_dict[i-1][3])
                data["words_aft_only"].append(data_dict[i+1][3])
                data["pos"].append(data_dict[i][5])                             
                data["upos"].append(data_dict[i][6]) 
                data["nentity"].append(data_dict[i][7]) 
                data["chunk"].append(data_dict[i][8]) 
                data["ptree"].append(data_dict[i][9]) 
                data["dep_pnum"].append(data_dict[i][10])   
                data["dep_label"].append(data_dict[i][11]) 
                data["sem_role"].append(data_dict[i][12][vn_index]) # index 12 stores dictionary if semantic roles of a sentence
                data["sem_role_v2"].append(self.get_sem_role_v2(data_dict[i][12][vn_index]))
                data["sem_role_v3"].append(self.get_sem_role_v3(data_dict[i][12][vn_index]))
                
                gold_srl = srl_dict[data_dict_info[i]][vn_index+1] #vn_index+1 because index 0 in srl_dict[tuple] represents verb, srl starts from index 1
                assert (len(gold_srl) == num_words)
                gold_srl_v2 = self.get_sem_role_v2(gold_srl)                
                data["sem_role_gold"].append( gold_srl )
                data["sem_role_gold_v2"].append(gold_srl_v2)
                
                gold_srl_grp, gold_srl_grp_cnt, focus_grp, verb_neg_grp, er_cnt = self.get_sem_role_grp(i, gold_srl_v2, data_dict[i][14], data_dict[i][13])  #index 14 stores focus, 13 stores verbal negation
                data["sem_role_gold_grp"].append(gold_srl_grp)
                data["sem_role_gold_grp_cnt"].append(gold_srl_grp_cnt)
                data["verb_neg"].append(data_dict[i][13]) 
                data["verb_neg_grp"].append(verb_neg_grp) 
                data["focus"].append(data_dict[i][14]) 
                data["focus_grp"].append(focus_grp) 
                sent_length.append(data_dict[i][0])
                pos += 1

        return data, sent_length 

    
    def get_actual_label(self, pred, sent_num, index, max_len):
        if index < max_len:
            if pred[sent_num][index] == "I_F": return "FOCUS"
            else: return "*"
        else: return "*"
    
    def create_new_obj_list(self, obj_list, pred, max_len, max_sent_size):
        newobj_list = copy.deepcopy(obj_list)        
        j = 0
        sent_num = 0
        corresp_sent_list = list(range(1,max_sent_size,3))
        corresp_indx = {e:i for i,e in enumerate(corresp_sent_list)}
        for i in range(len(obj_list)):            
            if sent_num in corresp_sent_list: 
                new_focus = self.get_actual_label(pred, corresp_indx[sent_num], j, max_len)
                newobj_list[i].focus = new_focus
            j += 1
            
            if (i == len(obj_list)-1) or ( i+1 < len(obj_list) and int(obj_list[i+1].word_num.strip()) == 1):
                j = 0
                sent_num += 1

        return newobj_list
    
    def print_to_file(self, obj_list, file_name):
        file_obj = open(file_name, "w")
        line = ''
        delim = " "
        for i in range(len(obj_list)):
            line = obj_list[i].col0 + delim + obj_list[i].col1 + delim +obj_list[i].col2 +  delim + obj_list[i].col3 + delim + obj_list[i].word + delim + obj_list[i].word_num + delim + obj_list[i].pos + delim + obj_list[i].name_entity + delim + obj_list[i].chunk + delim + obj_list[i].parse_tree + delim + obj_list[i].dep_pnum + delim + obj_list[i].dep_label  
            if len(obj_list[i].sem_role) == 0:
                line = line + delim + ""
            else:
                for elem in obj_list[i].sem_role:
                     line = line + delim + elem
            
            line = line + delim + obj_list[i].verbal_neg + delim + obj_list[i].focus
            
            file_obj.write(line)
            file_obj.write("\n")
                        
            if i+1 < len(obj_list)  and int(obj_list[i+1].word_num) == 1:                
                file_obj.write("\n")
        file_obj.close()    
    
    
class data_for_training():
    def unique_tokens(self, data_list_of_list, isLower = False):
        """
        Returns unique tokens from a list of list data
        :param data_list_of_list:
        :param isLower:
        :return:
        """
        token_all = set()
        for sentence in data_list_of_list:
            for token in sentence:
                if isLower == False:
                    token_all.add(token)
                else:
                    token_all.add(token.lower())
        
        return list(token_all)   

    def get_unique_tokens(self, data, isLower):
        """
        Returns dictionary of data of unique tokens.
        :param data:
        :param isLower:
        :return:
        """
        token_dict = {}         
        
        if "words" in  data: token_dict["words"] = self.unique_tokens(data["words"]+data["words_bef_only"]+data["words_aft_only"], isLower)   # For sentences in the data
        if "pos" in  data: token_dict["pos"] = self.unique_tokens(data["pos"])
        if "upos" in  data: token_dict["upos"] = self.unique_tokens(data["upos"])
        if "nentity" in  data: token_dict["nentity"] = self.unique_tokens(data["nentity"])
        if "chunk" in  data: token_dict["chunk"] = self.unique_tokens(data["chunk"])
        if "ptree" in  data: token_dict["ptree"] = self.unique_tokens(data["ptree"])
        if "dep_pnum" in  data: token_dict["dep_pnum"] = self.unique_tokens(data["dep_pnum"])
        if "dep_label" in  data: token_dict["dep_label"] = self.unique_tokens(data["dep_label"])
        if "sem_role" in  data: token_dict["sem_role"] = self.unique_tokens(data["sem_role"])
        if "sem_role_v2" in  data: token_dict["sem_role_v2"] = self.unique_tokens(data["sem_role_v2"])
        if "sem_role_v3" in  data: token_dict["sem_role_v3"] = self.unique_tokens(data["sem_role_v3"])
        if "sem_role_gold" in  data: token_dict["sem_role_gold"] = self.unique_tokens(data["sem_role_gold"])
        if "sem_role_gold_v2" in  data: token_dict["sem_role_gold_v2"] = self.unique_tokens(data["sem_role_gold_v2"])
        if "sem_role_gold_grp" in  data: token_dict["sem_role_gold_grp"] = self.unique_tokens(data["sem_role_gold_grp"])
        if "verb_neg" in  data: token_dict["verb_neg"] = self.unique_tokens(data["verb_neg"])
        if "verb_neg_grp" in  data: token_dict["verb_neg_grp"] = self.unique_tokens(data["verb_neg_grp"])
        if "focus" in  data: token_dict["focus"] = self.unique_tokens(data["focus"])           
        if "focus_grp" in  data: token_dict["focus_grp"] = self.unique_tokens(data["focus_grp"])           
        
        return token_dict    

    def token_indexing(self, token_list, isPad =True, isUnknown=True):
        count = 0
        if isPad == True: count += 1
        if isUnknown == True: count += 1        
        
        token2index = {w:i+count for i, w in enumerate(token_list)} 
        if isPad == True: token2index[PAD] = 0
        if isUnknown == True: token2index[UNKNOWN] = 1 if isPad == True else 0
        
        index2token = {i:w for w, i in token2index.items()}        
        return token2index, index2token

    def get_indexing(self, token_dict):
        index_dict = {}
        if "words" in token_dict: 
            isPad =True; isUnknown=True
            token2index, index2token = self.token_indexing(token_dict["words"], isPad, isUnknown)
            index_dict["word2index"] = token2index; index_dict["index2word"] = index2token
            
        if "pos" in token_dict: 
            isPad =True; isUnknown=True
            token2index, index2token = self.token_indexing(token_dict["pos"], isPad, isUnknown)
            index_dict["pos2index"] = token2index; index_dict["index2pos"] = index2token
            
        if "upos" in token_dict: 
            isPad =True; isUnknown=True
            token2index, index2token = self.token_indexing(token_dict["upos"], isPad, isUnknown)
            index_dict["upos2index"] = token2index; index_dict["index2upos"] = index2token                    
        
        if "nentity" in token_dict: 
            isPad =True; isUnknown=True
            token2index, index2token = self.token_indexing(token_dict["nentity"], isPad, isUnknown)
            index_dict["nentity2index"] = token2index; index_dict["index2nentity"] = index2token
            
        if "chunk" in token_dict: 
            isPad =True; isUnknown=True
            token2index, index2token = self.token_indexing(token_dict["chunk"], isPad, isUnknown)
            index_dict["chunk2index"] = token2index; index_dict["index2chunk"] = index2token
            
        if "ptree" in token_dict: 
            isPad =True; isUnknown=True
            token2index, index2token = self.token_indexing(token_dict["ptree"], isPad, isUnknown)
            index_dict["ptree2index"] = token2index; index_dict["index2ptree"] = index2token
            
        if "dep_pnum" in token_dict: 
            isPad =True; isUnknown=True
            token2index, index2token = self.token_indexing(token_dict["dep_pnum"], isPad, isUnknown)
            index_dict["dep_pnum2index"] = token2index; index_dict["index2dep_pnum"] = index2token
            
        if "dep_label" in token_dict: 
            isPad =True; isUnknown=True
            token2index, index2token = self.token_indexing(token_dict["dep_label"], isPad, isUnknown)
            index_dict["dep_label2index"] = token2index; index_dict["index2dep_label"] = index2token
        
        if "sem_role" in token_dict: 
            isPad =True; isUnknown=True
            token2index, index2token = self.token_indexing(token_dict["sem_role"], isPad, isUnknown)
            index_dict["sem_role2index"] = token2index; index_dict["index2sem_role"] = index2token
        
        if "sem_role_v2" in token_dict: 
            isPad =True; isUnknown=True
            token2index, index2token = self.token_indexing(token_dict["sem_role_v2"], isPad, isUnknown)
            index_dict["sem_role_v22index"] = token2index; index_dict["index2sem_role_v2"] = index2token  
            
        if "sem_role_v3" in token_dict: 
            isPad =True; isUnknown=True
            token2index, index2token = self.token_indexing(token_dict["sem_role_v3"], isPad, isUnknown)
            index_dict["sem_role_v32index"] = token2index; index_dict["index2sem_role_v3"] = index2token 

        if "sem_role_gold" in token_dict: 
            isPad =True; isUnknown=True
            token2index, index2token = self.token_indexing(token_dict["sem_role_gold"], isPad, isUnknown)
            index_dict["sem_role_gold2index"] = token2index; index_dict["index2sem_role_gold"] = index2token  

        if "sem_role_gold_v2" in token_dict: 
            isPad =True; isUnknown=True
            token2index, index2token = self.token_indexing(token_dict["sem_role_gold_v2"], isPad, isUnknown)
            index_dict["sem_role_gold_v22index"] = token2index; index_dict["index2sem_role_gold_v2"] = index2token              
            
        if "sem_role_gold_grp" in token_dict: 
            isPad =True; isUnknown=True
            token2index, index2token = self.token_indexing(token_dict["sem_role_gold_grp"], isPad, isUnknown)
            index_dict["sem_role_gold_grp2index"] = token2index; index_dict["index2sem_role_gold_grp"] = index2token
            
        if "verb_neg" in token_dict: 
            isPad =True; isUnknown=False
            token2index, index2token = self.token_indexing(token_dict["verb_neg"], isPad, isUnknown)
            index_dict["verb_neg2index"] = token2index; index_dict["index2verb_neg"] = index2token    

        if "verb_neg_grp" in token_dict: 
            isPad =True; isUnknown=False
            token2index, index2token = self.token_indexing(token_dict["verb_neg_grp"], isPad, isUnknown)
            index_dict["verb_neg_grp2index"] = token2index; index_dict["index2verb_neg_grp"] = index2token    

                        
        if "focus" in token_dict: 
            isPad =True; isUnknown=False
            token2index, index2token = self.token_indexing(token_dict["focus"], isPad, isUnknown)
            index_dict["focus2index"] = token2index; index_dict["index2focus"] = index2token 
            
        if "focus_grp" in token_dict: 
            isPad =True; isUnknown=False
            token2index, index2token = self.token_indexing(token_dict["focus_grp"], isPad, isUnknown)
            index_dict["focus_grp2index"] = token2index; index_dict["index2focus_grp"] = index2token 
        
        return index_dict

    def get_sent_with_padding(self, sentence_list, unique_token_list, token2index, max_len, isLower = False):
        """
        Returns the list of list of sentences with indicies. Each token of a sentence is replaced with it's corresponding index number.
        :param sentence_list:
        :param unique_token_list:
        :param token2index:
        :param max_len:
        :param isLower:
        :return:
        """
        indexed_sent_list = []
        for sentence in sentence_list:
            token_list = []
            for token in sentence:
                if isLower == True: token = token.lower()
                if token in unique_token_list:
                    token_list.append(token2index[token] )
                else:
                    token_list.append(token2index[UNKNOWN])  #for unknown word
            indexed_sent_list.append(token_list)
            token_list = []
        
        indexed_sent_list = pad_sequences(maxlen=max_len, sequences=indexed_sent_list, padding="post", value=token2index[PAD])
        return indexed_sent_list
    

    def get_processed_data(self, max_len, index_dict, token_dict, data, isLower = False):
        """
        Prepare indexed and padded tokens for all the data.
        :param max_len:
        :param index_dict:
        :param token_dict:
        :param data:
        :param isLower:
        :return:
        """
        processed_data = {}
        
        if "words" in  data: 
            processed_data["words"] = self.get_sent_with_padding(data["words"], token_dict["words"], index_dict["word2index"], max_len, isLower)              
        if "words_bef" in  data: 
            processed_data["words_bef"] = self.get_sent_with_padding(data["words_bef"], token_dict["words"], index_dict["word2index"], max_len, isLower)              
        if "words_aft" in  data: 
            processed_data["words_aft"] = self.get_sent_with_padding(data["words_aft"], token_dict["words"], index_dict["word2index"], max_len, isLower)              
        if "words_bef_aft" in  data: 
            processed_data["words_bef_aft"] = self.get_sent_with_padding(data["words_bef_aft"], token_dict["words"], index_dict["word2index"], max_len, isLower)              
        if "words_bef_only" in  data: 
            processed_data["words_bef_only"] = self.get_sent_with_padding(data["words_bef_only"], token_dict["words"], index_dict["word2index"], max_len, isLower)              
        if "words_aft_only" in  data: 
            processed_data["words_aft_only"] = self.get_sent_with_padding(data["words_aft_only"], token_dict["words"], index_dict["word2index"], max_len, isLower)              
        
        if "pos" in  data: 
            processed_data["pos"] = self.get_sent_with_padding(data["pos"], token_dict["pos"], index_dict["pos2index"], max_len)
        if "upos" in  data: 
            processed_data["upos"] = self.get_sent_with_padding(data["upos"], token_dict["upos"], index_dict["upos2index"], max_len)
        if "nentity" in  data: 
            processed_data["nentity"] = self.get_sent_with_padding(data["nentity"], token_dict["nentity"], index_dict["nentity2index"], max_len)
        if "chunk" in  data: 
            processed_data["chunk"] = self.get_sent_with_padding(data["chunk"], token_dict["chunk"], index_dict["chunk2index"], max_len)
        if "ptree" in  data: 
            processed_data["ptree"] = self.get_sent_with_padding(data["ptree"], token_dict["ptree"], index_dict["ptree2index"], max_len)            
        if "dep_pnum" in  data: 
            processed_data["dep_pnum"] = self.get_sent_with_padding(data["dep_pnum"], token_dict["dep_pnum"], index_dict["dep_pnum2index"], max_len)
        if "dep_label" in  data: 
            processed_data["dep_label"] = self.get_sent_with_padding(data["dep_label"], token_dict["dep_label"], index_dict["dep_label2index"], max_len)
        if "sem_role" in  data: 
            processed_data["sem_role"] = self.get_sent_with_padding(data["sem_role"], token_dict["sem_role"], index_dict["sem_role2index"], max_len)
        if "sem_role_v2" in  data: 
            processed_data["sem_role_v2"] = self.get_sent_with_padding(data["sem_role_v2"], token_dict["sem_role_v2"], index_dict["sem_role_v22index"], max_len)
        if "sem_role_v3" in  data: 
            processed_data["sem_role_v3"] = self.get_sent_with_padding(data["sem_role_v3"], token_dict["sem_role_v3"], index_dict["sem_role_v32index"], max_len)
        if "sem_role_gold" in  data: 
            processed_data["sem_role_gold"] = self.get_sent_with_padding(data["sem_role_gold"], token_dict["sem_role_gold"], index_dict["sem_role_gold2index"], max_len)
        if "sem_role_gold_v2" in  data: 
            processed_data["sem_role_gold_v2"] = self.get_sent_with_padding(data["sem_role_gold_v2"], token_dict["sem_role_gold_v2"], index_dict["sem_role_gold_v22index"], max_len)            
        if "sem_role_gold_grp" in  data: 
            processed_data["sem_role_gold_grp"] = self.get_sent_with_padding(data["sem_role_gold_grp"], token_dict["sem_role_gold_grp"], index_dict["sem_role_gold_grp2index"], max_len)                            
        if "verb_neg" in  data: 
            processed_data["verb_neg"] = self.get_sent_with_padding(data["verb_neg"], token_dict["verb_neg"], index_dict["verb_neg2index"], max_len)
        if "verb_neg_grp" in  data: 
            processed_data["verb_neg_grp"] = self.get_sent_with_padding(data["verb_neg_grp"], token_dict["verb_neg_grp"], index_dict["verb_neg_grp2index"], max_len)
        if "focus" in  data: 
            processed_data["focus"] = self.get_sent_with_padding(data["focus"], token_dict["focus"], index_dict["focus2index"], max_len)   
        if "focus_grp" in  data: 
            processed_data["focus_grp"] = self.get_sent_with_padding(data["focus_grp"], token_dict["focus_grp"], index_dict["focus_grp2index"], max_len)

        return processed_data

    def get_labels(self, labels, unique_labels):
        labels = [to_categorical(l, num_classes = unique_labels) for l in  labels] 
        return np.array(labels)

    def get_data_for_training(self, file_name, max_len, srl_dict, isLower, label_name):
        """
        prepare final dataset and lables for training.
        :param file_name:
        :param max_len:
        :param srl_dict:
        :param isLower:
        :param label_name:
        :return:
        """
        file_obj = open(file_name, "r")
        print("1. Started reading data--------------------")
        dp_obj = data_preparation()
        tr_prep_obj = dp_obj.data_load(file_obj)
        
        print("2. Started extracting data for focus detection--------------------")
        detail_data_dict, data_dict_info  = dp_obj.get_data_details(tr_prep_obj)
        data_for_focus, sent_length = dp_obj.data_for_focus_resolution(detail_data_dict, data_dict_info, srl_dict)
        
        print("3. Started extracting unique tokens of the features/labels--------------------")
        token_dict = self.get_unique_tokens(data_for_focus, isLower)
        
        print("4. Started indexing the tokens of the features/labels--------------------")
        index_dict = self.get_indexing(token_dict)
        
        print("5. Started preparing data with padding and indexing for training--------------------")
        proc_data = self.get_processed_data(max_len, index_dict, token_dict, data_for_focus, isLower)
               
        print("6. Started preparing labels with padding and indexing for training--------------------")
        num_labels = len(token_dict[label_name]) + 1 #+1 for padding
        labels = self.get_labels(proc_data[label_name], num_labels)
        
        # Extention.Getting original for ELMo
        proc_data["words_org"]=data_for_focus["words"]
        proc_data["words_bef_only_org"]=data_for_focus["words_bef_only"]
        proc_data["words_aft_only_org"]=data_for_focus["words_aft_only"]

        
        print("7. Data and labels preparation completed.!!")                
        return proc_data, labels, token_dict, index_dict, sent_length, data_for_focus["sem_role_gold_grp_cnt"]   

    def get_data_for_validation(self, file_name, max_len, index_dict, token_dict, srl_dict, isLower, label_name):
        """
        prepare final data and lables for validation dataset
        :param file_name:
        :param max_len:
        :param index_dict:
        :param token_dict:
        :param srl_dict:
        :param isLower:
        :param label_name:
        :return:
        """
        file_obj = open(file_name, "r")    
        
        print("1. Started reading data--------------------")
        dp_obj = data_preparation()
        tr_prep_obj = dp_obj.data_load(file_obj)
        
        print("2. Started extracting data for focus detection--------------------")
        detail_data_dict, data_dict_info = dp_obj.get_data_details(tr_prep_obj)
        data_for_focus, sent_length = dp_obj.data_for_focus_resolution(detail_data_dict, data_dict_info, srl_dict)
        
        print("3. Started preparing data with padding and indexing for evaluation--------------------")
        proc_data = self.get_processed_data(max_len, index_dict, token_dict, data_for_focus, isLower)
        
        print("4. Started generating labels with padding and indexing for evaluation--------------------")       
        num_labels = len(token_dict[label_name]) + 1 #+1 for padding
        labels = self.get_labels(proc_data[label_name], num_labels)    
        
        
        # Extention.Getting original for ELMo
        proc_data["words_org"]=data_for_focus["words"]
        proc_data["words_bef_only_org"]=data_for_focus["words_bef_only"]
        proc_data["words_aft_only_org"]=data_for_focus["words_aft_only"]
        
        print("5. Data and labels preparation completed.!!")
        return proc_data, labels, sent_length, data_for_focus["sem_role_gold_grp_cnt"]
    
    
    def prepare_training_data(self, data, features_dict, index_dict, embed_dim=300):
        x = []  
        num_tokens = {}
        embed_dims = {}
        if "words" in features_dict: 
            x.append(data["words"]) 
            num_tokens["words"] = len(index_dict["word2index"])
            embed_dims["words"] = embed_dim
            
        if "words_bef" in features_dict: 
            x.append(data["words_bef"]) 
            num_tokens["words_bef"] = len(index_dict["words_bef2index"])
            embed_dims["words_bef"] = embed_dim  
            
        if "words_aft" in features_dict: 
            x.append(data["words_aft"]) 
            num_tokens["words_aft"] = len(index_dict["words_aft2index"])
            embed_dims["words_aft"] = embed_dim            
            
        if "words_bef_aft" in features_dict: 
            x.append(data["words_bef_aft"]) 
            num_tokens["words_bef_aft"] = len(index_dict["words_bef_aft2index"])
            embed_dims["words_bef_aft"] = embed_dim  
                    
        if "pos" in features_dict: 
            x.append(data["pos"])
            num_tokens["pos"] = len(index_dict["pos2index"])
            embed_dims["pos"] = embed_dim
            
        if "upos" in features_dict: 
            x.append(data["upos"])
            num_tokens["upos"] = len(index_dict["upos2index"])
            embed_dims["upos"] = embed_dim
            
        if "nentity" in features_dict: 
            x.append(data["nentity"])
            num_tokens["nentity"] = len(index_dict["nentity2index"])
            embed_dims["nentity"] = embed_dim
            
        if "chunk" in features_dict: 
            x.append(data["chunk"])
            num_tokens["chunk"] = len(index_dict["chunk2index"])
            embed_dims["chunk"] = embed_dim
            
        if "ptree" in features_dict: 
            x.append(data["ptree"])
            num_tokens["ptree"] = len(index_dict["ptree2index"])
            embed_dims["ptree"] = embed_dim
            
        if "dep_pnum" in features_dict: 
            x.append(data["dep_pnum"])
            num_tokens["dep_pnum"] = len(index_dict["dep_pnum2index"])
            embed_dims["dep_pnum"] = embed_dim
            
        if "dep_label" in features_dict: 
            x.append(data["dep_label"])
            num_tokens["dep_label"] = len(index_dict["dep_label2index"])
            embed_dims["dep_label"] = embed_dim   
            
        if "sem_role" in features_dict: 
            x.append(data["sem_role"])
            num_tokens["sem_role"] = len(index_dict["sem_role2index"])
            embed_dims["sem_role"] = 200
            
        if "sem_role_v2" in features_dict: 
            x.append(data["sem_role_v2"])
            num_tokens["sem_role_v2"] = len(index_dict["sem_role_v22index"])
            embed_dims["sem_role_v2"] = 200  
            
        if "sem_role_v3" in features_dict: 
            x.append(data["sem_role_v3"])
            num_tokens["sem_role_v3"] = len(index_dict["sem_role_v32index"])
            embed_dims["sem_role_v3"] = 50
            
        if "sem_role_gold" in features_dict: 
            x.append(data["sem_role_gold"])
            num_tokens["sem_role_gold"] = len(index_dict["sem_role_gold2index"])
            embed_dims["sem_role_gold"] = 200
            
        if "sem_role_gold_v2" in features_dict: 
            x.append(data["sem_role_gold_v2"])
            num_tokens["sem_role_gold_v2"] = len(index_dict["sem_role_gold_v22index"])
            embed_dims["sem_role_gold_v2"] = 200
            
        if "sem_role_gold_grp" in features_dict: 
            x.append(data["sem_role_gold_grp"])
            num_tokens["sem_role_gold_grp"] = len(index_dict["sem_role_gold_grp2index"])
            embed_dims["sem_role_gold_grp"] = 200
            
        if "verb_neg" in features_dict: 
            x.append(data["verb_neg"])
            num_tokens["verb_neg"] = len(index_dict["verb_neg2index"])
            embed_dims["verb_neg"] = 50   
            
        if "verb_neg_grp" in features_dict: 
            x.append(data["verb_neg_grp"])
            num_tokens["verb_neg_grp"] = len(index_dict["verb_neg_grp2index"])
            embed_dims["verb_neg_grp"] = 50  
            
        if "words_bef_only" in features_dict: 
            x.append(data["words_bef_only"]) 
            num_tokens["words_bef_only"] = len(index_dict["words_bef_only2index"])
            embed_dims["words_bef_only"] = embed_dim  

        if "words_aft_only" in features_dict: 
            x.append(data["words_aft_only"]) 
            num_tokens["words_aft_only"] = len(index_dict["words_aft_only2index"])
            embed_dims["words_aft_only"] = embed_dim                
                                    
        return x, num_tokens, embed_dims
    
        
    def prepare_training_data_grp(self, proc_data, y, sr_grp_cnt, num_srole, srole_each_cnt, max_len):
        rows, cols = proc_data["words"].shape
        data_wrd = np.zeros((rows, num_srole, srole_each_cnt), dtype=int)       
        data_srg = np.zeros((rows, num_srole), dtype=int)
        data_srg_vb_neg = np.zeros((rows, num_srole), dtype=int)
        y = np.argmax(y, axis=2) #3D to 2D
        data_y = np.zeros((rows, num_srole), dtype=int)    
        for i in range(rows):
            end = 0
            for j in range( np.min([num_srole, len(sr_grp_cnt[i])])):
                #preparing word data
                start = end
                end = start + sr_grp_cnt[i][j]
                d = np.min( [end - start, srole_each_cnt]) # taking the minimum. limiting end-start
                d = np.min( [d, max_len-start] ) #limitting max_len-start
                if start < max_len-1:
                    data_wrd[i,j,0:d] = proc_data["words"][i][start:start+d]
                    
                    # semantic role data
                    data_srg[i,j] = proc_data["sem_role_gold_grp"][i,j]
                    
                    # Verbal neg
                    data_srg_vb_neg[i,j] = proc_data["verb_neg_grp"][i,j]
                    
                    
                    # target
                    data_y[i,j] = y[i,j]
                else:
                    break
                
        data_wrd_r = np.reshape(data_wrd, (rows, num_srole*srole_each_cnt))  
        proc_data["sem_role_gg_words"] = data_wrd_r           #dim rows x (num_srole*srole_each_cnt)    
        proc_data["sem_role_gold_grp_less"] = data_srg        #dim rows x num_srole
        proc_data["verb_neg_grp_less"] = data_srg_vb_neg      #dim rows x num_srole
        proc_data["focus_grp_less"] = data_y                  #dim rows x num_srole
        return proc_data
    
    def prepare_y_classify(self, y, sem_role, index2label):
        rows, cols = y.shape
        y = [[index2label[ind] for ind in sent] for sent in y] #converting a index to it's actual label name
        #sem_role = [[index2sem_role[ind] for ind in sent] for sent in sem_role] #converting a index to it's actual label name
        class_labels = []
        
        for i in range(rows):
            isEntered = False
            for j in range(cols):
                if y[i][j] == "I_F":
                    class_labels.append(sem_role[i][j])
                    isEntered = True
                    break
            if not isEntered:
                class_labels.append(1) #1 for Unknown, here also for no I_F found case.
            
        return np.array(class_labels)
    
    def focus_to_role_label(self, y, sem_role, index2label, index2sem_role):
        """
        This function tags role label as focus for each instance.
        Args:
            y: (list of list), each element of a list of list contans mapped index for label (e.g. 0:I_F).
            sem_role: (list of list), sequence of semantic roles(golds from propbank, processed roles(e.g A0: [(A0*, *)] = [A0, A0] ) ) for the targetted negated verb.
            index2label: (dict), index to label mapping.
            index2sem_role: (dict), index to semantic role (gold and processed) mapping.
        Output:
            role_labels: (np.array), Role label of a focus 
        """
        rows, cols = y.shape
        y = [[index2label[ind] for ind in sent] for sent in y] #converting a index to it's actual label name
        sem_role = [[index2sem_role[ind] for ind in sent] for sent in sem_role] #converting a index to it's actual label name
        class_labels = []
        
        for i in range(rows):
            isEntered = False
            for j in range(cols):
                if y[i][j] == "I_F":
                    class_labels.append(sem_role[i][j])
                    isEntered = True
                    break
            if not isEntered:
                print("No focus is tagged for index {}".format(i))
                class_labels.append("NA") #1 for Unknown, here also for no I_F found case.
            
        return class_labels
    
    def tag_sematic_roles(self, proc_data, token2index, data_name):
        sem_roles = ['(A0*)', '(A1*)', '(A2*)', '(A3*)', '(A4*)', '(AM-NEG*)', '(AM-LOC*)', '(V*)', '(AM-MOD*)', '(AM-TMP*)', '(AM-MNR*)', '(AM-EXT*)', '(AM-CAU*)', '(AM-PRD*)', '(AM-DIR*)', '(AM-ADV*)', '(AM-DIS*)', '(AM-PNC*)']
        num_actual_semr = len(sem_roles)
        
        
        rows, _ = proc_data[data_name].shape
        sr_tag = np.zeros((rows, num_actual_semr), dtype=int)
        
        for i in range(rows):
            for j in range(num_actual_semr):
                if token2index[sem_roles[j]] in proc_data[data_name][i]:
                    sr_tag[i][j] = 1.0
        proc_data["sem_role_static_tag"] = sr_tag
        return proc_data, num_actual_semr
    
    
    def train_dev_split(self, proc_data, tr_y, val_percent, isSeq2seq):
        data_size = len(proc_data["words"])
        data_indices = list(range(data_size))
        
        x_train_sam, x_val_sam, _, _ = train_test_split( data_indices, data_indices, test_size=val_percent, random_state=42) #random sampling
        tr_proc_data  = {}
        val_proc_data = {}
        for key, value in proc_data.items():
            if key in ["words_org", "words_bef_only_org", "words_aft_only_org"]:
                tr_proc_data[key]  = list(np.array(value)[x_train_sam])
                val_proc_data[key] = list(np.array(value)[x_val_sam])
            else:
                tr_proc_data[key]  = value[x_train_sam]
                val_proc_data[key] = value[x_val_sam]
            
        
        train_y = tr_y[x_train_sam]
        val_y   = tr_y[x_val_sam]
        
        if not isSeq2seq: #only for classification case
            print("total samples: {}, train samples: {}, dev samples: {}, common samples: {}".format(data_size, len(x_train_sam), len(x_val_sam), len( set(x_train_sam).intersection(set(x_val_sam))   ) ))         
            print("train len: {}, val len: {}, train uniq: {}, val unique: {}".format(len(tr_proc_data["words"]), len(val_proc_data["words"]), len(np.unique(train_y)), len(np.unique(val_y)) ) )
            for e in np.unique(train_y): 
                print("sr: {}, count: {}".format(e, list(train_y).count(e)))
            for e in np.unique(val_y): 
                print("sr: {}, count: {}".format(e, list(val_y).count(e))) 
        
        
        return tr_proc_data, train_y, val_proc_data, val_y
            
    def get_original(self, data, index2token, token2index):
        pad_number = token2index['PAD']
        new_data = []
        rows, cols = data.shape
        
        for i in range(rows):
            sequence = []
            for j in range(cols):
                if data[i][j] != pad_number:
                    sequence.append( index2token[ data[i][j] ] )            
            if len(sequence) == 0:
                print("i: {}, j: {}, pad_number: {}, seq: {}".format(i,j,pad_number, data[i]) )
                assert len(sequence) == 0
            new_data.append(sequence)
        return new_data
    
    def get_data_for_scope(self, data, token_dict, index_dict, isCleanup=False):
        data_dict_scope = defaultdict(list)
        for s in data["words"]:
            data_dict_scope["words"].append([ index_dict["index2word"][w] for w in s if w !=index_dict["word2index"]["PAD"] ])
        
        for s in data["sem_role_gold"]: #or use sem_role_gold_V2
            sem_role_gold= [ index_dict["index2sem_role_gold"][w] for w in s if w !=index_dict["sem_role_gold2index"]["PAD"] ]
            data_dict_scope["cues"].append( ["I_C" if role == "(AM-NEG*)" else "O_C" for role in sem_role_gold] )
        
        for s in data["upos"]:
            data_dict_scope["upos"].append([ index_dict["index2upos"][w] for w in s if w != index_dict["upos2index"]["PAD"] ])


        # removing extra punctuation (Spacy's mistake)
        if isCleanup:
            for r in range( len(data_dict_scope["words"]) ):
                sent = data_dict_scope["words"][r]
                upos_sent = data_dict_scope["upos"][r]
                for i in range( len(sent) ):                                
                    if (sent[i] == "``") or (sent[i] == "''")  or (sent[i] == "--"): #removing double punctuations in case of these found
                        if i+1 < len(upos_sent) and upos_sent[i+1] == "PUNCT":
                            data_dict_scope["upos"][r].pop(i+1)
                    if "-" in sent[i]:  #words e.g "city-type". making one token instead of three
                        if i+2 < len(upos_sent) and upos_sent[i+1] == "PUNCT":
                            data_dict_scope["upos"][r].pop(i+1)
                            data_dict_scope["upos"][r].pop(i+2)
                    
                    # applying this again if in case miss someting       
                    if (sent[i] == "``") or (sent[i] == "''")  or (sent[i] == "--"):
                        if i+1 < len(upos_sent) and upos_sent[i+1] == "PUNCT":
                            data_dict_scope["upos"][r].pop(i+1)
                                
        return  data_dict_scope     

    
    def integrate_scope_data(self, data, prediction, phase, token_dict, index_dict, max_len, token_count_list=None):
        if phase == "training":
            token_dict["scope"] = ['I_S', 'O_S']
            index_dict["index2scope"] = {0:'PAD', 1:'O_S', 2:'I_S'}
            index_dict["scope2index"] = {val:key for key,val in index_dict["index2scope"].items()}
        data["scope"] = self.get_sent_with_padding(prediction, token_dict["scope"], index_dict["scope2index"], max_len)                
        
        return data, token_dict, index_dict


class pb_sem_role():
    def get_srl(self, file, section, srl_dict):
        var_dict = defaultdict(list)
        sent_num = 0
        for line in file:            
            tokens = line.split()
            if len(tokens) > 0:                         
                for i in range(len(tokens)): # intex 0 of teokens is a verb or "-". Semantic role starts with index 1
                    var_dict[i].append(tokens[i])
            else:
                srl_dict[(section, str(sent_num)) ] =  var_dict
                sent_num += 1
                var_dict = defaultdict(list)
        print("Number of sentences (cumulative): {}".format(len(srl_dict)))
                        
    
    def get_gold_srl(self, dir_name):
        srl_dict = {}
        for file in os.listdir(dir_name):
            try:
                if file.endswith(".props"):
                    section = file[-8:-6]
                    print("Collecting gold SRL for the file {}.".format(file) )
                    file = open(dir_name + "/" + file, "r")
                    self.get_srl(file, section, srl_dict)
                    file.close()
            except Exception as err_msg:
                raise err_msg
                print("Read error.")
                
        return srl_dict


