# -*- coding: utf-8 -*-
"""
@author: Md Mosharaf Hossain
Doctoral Student in CSE at the University of North Texas
"""

import numpy as np
BATCH = 124

class evaluate_model():
    def predict_multihop(self, model_obj, test_x, sent_lens):
        """
        Predict a test dataset in multi-hop.
        :param model_obj:
        :param test_x:
        :param sent_lens:
        :return:
        """
        rows = len(test_x[0])
        num_steps = int(np.ceil(rows/BATCH))
        for i in range(num_steps):
            start = i*BATCH
            end   = (i+1)*BATCH if i < num_steps-1 else rows
            part_test_x = {}

            part_lens = list(np.array(sent_lens)[start:end])
            for indx in test_x:
                part_test_x[indx] = test_x[indx][start:end]

            if i == 0:
                outputs,_,_ = model_obj.model(part_test_x, part_lens)
            else:
                temp_outputs,_,_ = model_obj.model(part_test_x, part_lens)
                outputs = outputs+temp_outputs
        
        return outputs
                        
                    
    
    def predict(self, model_obj, test_x, sent_lens):
        """
        Predict a dataset.
        :param model_obj:
        :param test_x:
        :param sent_lens:
        :return:
        """
        outputs = self.predict_multihop(model_obj, test_x, sent_lens) # outputs dim: batch_size x seq_len
        return outputs
    
    def get_loss(self, model_obj, test_x, test_y, sent_lens):
        """
        Calculate the loss.
        :param model_obj:
        :param test_x:
        :param test_y:
        :param sent_lens:
        :return:
        """
        if not model_obj.isCRF:
            outputs = self.predict_multihop(model_obj, test_x, sent_lens)
            loss = model_obj.loss_fn(outputs, test_y)  
        else: #CRF loss
            #return 0.0 # for the time being to avoid cuda error. work on it later.
            _, emissions, crf = model_obj.model(test_x, sent_lens)
            loss = -1.0 * crf(emissions, test_y) #negative log likelihood
            loss = loss/len(test_y) #average loss
        
        loss = np.float(loss.data.cpu().numpy())  #bring the loss back to the cpu and convert to numpy            
        return loss 
        
    
    def get_eval(self, model_obj, test_x, test_y, sent_lens, index2token, label_name, pad_name):
        """
        Get the evaluation scores.
        :param model_obj:
        :param test_x:
        :param test_y:
        :param sent_lens:
        :param index2token:
        :param label_name:
        :param pad_name:
        :return:
        """

        # set model to evaluation mode. Some layers (e.g BatchNorm, Dropout) have different behavior during training and evaluation. So, it is important to set the mode
        model_obj.model.eval()
        loss = self.get_loss(model_obj, test_x, test_y, sent_lens)
        
        
        # actual labels _____________________________________________________________________
        rows, cols = test_x[model_obj.sem_rl_indx].shape  # get the dimension, rows = batch_size, cols = seq_len
        test_y = test_y.data.cpu().numpy() # bring test_y back to cpu and then convert to numpy array if the calculation is done in GPU 

        test_y = [[index2token[ind] for ind in sent] for sent in test_y] #converting a index to it's actual label name
        outputs = self.predict(model_obj, test_x, sent_lens)
        if not model_obj.isCRF:
            outputs = np.reshape(outputs, (rows, cols)) # dim batch_size x seq_len
        outputs = [[index2token[ind] for ind in sent] for sent in outputs] #converting a index to it's actual label name

        
        # Measure the scores
        true_positives = 0      # if system and gold shows positive
        predicted_positives = 0 # all predicted positives
        actual_positives = 0    # gold positives 
        error_indices = []
        for i in range(rows):
            correct = 1
            p_label = 0
            g_label = 0
            for j in range(cols):
                if test_y[i][j] == pad_name:
                    #print("test_y: {}".format(test_y[i]))
                    #print("outputs: {}".format(outputs[i]))
                    break
                else:
                    if test_y[i][j] == label_name:
                        g_label = 1
                    
                    if outputs[i][j] == label_name:
                        p_label = 1
                    
                    if test_y[i][j] != outputs[i][j]:
                        correct = 0
            if p_label == 0 or g_label == 0:
                correct = 0
            
            if correct == 0:
                error_indices.append(i)
            true_positives      += correct
            predicted_positives += p_label
            actual_positives    += g_label
        
        precision = true_positives/predicted_positives if predicted_positives > 0 else 0.0
        recall    = true_positives/actual_positives if actual_positives > 0 else 0.0
        F1        = (2* precision *recall)/(precision + recall) if (precision + recall) > 0 else 0.0
        eval_dict = {"true_positives": true_positives, "predicted_positives": predicted_positives, "actual_positives": actual_positives, "precision": precision, "recall": recall, "F1":F1, "loss": loss}
        return eval_dict, error_indices, outputs, test_y
    

    
    def predict_test_single(self, model_obj, test_x, index2label, rows, cols, sent_lens):
        """
        Get predicted labels.
        :param model_obj:
        :param test_x:
        :param index2label:
        :param rows:
        :param cols:
        :param sent_lens:
        :return:
        """
        # set model to evaluation mode. Some layers (e.g BatchNorm, Dropout) have different behavior during training and evaluation. So, it is important to set the mode
        model_obj.model.eval()
        test_pred = self.predict(model_obj, test_x, sent_lens)
        if not model_obj.isCRF:
            test_pred = np.reshape(test_pred, (rows, cols)) # Dim: batch_size x seq_len
        test_pred = [[index2label[ind] for ind in sent] for sent in test_pred] #converting a index to it's actual label name

        return test_pred
            
            
            
