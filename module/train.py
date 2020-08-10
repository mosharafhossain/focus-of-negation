# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 00:11:43 2019

@author: Md Mosharaf Hossain
"""
import module.evaluate as evaluate
import module.batch as batch
import numpy as np
import torch
import tqdm


class model_train():
    def train(self, epoch, model_obj, data_iterator, num_steps, progress, data_eval=None):
        """
        Train the neural model on num_steps batches of each epoch.
        :param epoch:
        :param model_obj:
        :param data_iterator:
        :param num_steps:
        :param progress:
        :param data_eval:
        :return:
        """
            
        for batch_id in range(num_steps):
            # set model to training mode. This setting is necessary if the model is also set to evaluation mode at some point
            model_obj.model.train()   
            
            # fetch the next training batch
            train_batch, labels_batch, batch_lens = next(data_iterator)
    
            # compute model output and loss
            if not model_obj.isCRF:
                output_batch = model_obj.model(train_batch, batch_lens)
                loss = model_obj.loss_fn(output_batch, labels_batch)
                print("loss: {}".format(loss))
            else: #CRF loss
                _, emissions, crf = model_obj.model(train_batch, batch_lens)
                loss = -1.0 * crf(emissions, labels_batch) #negative log likelihood, need to explore more on this
                loss = loss/len(labels_batch) #average loss                                    
            
            # clear previous gradients, 
            model_obj.optimizer.zero_grad()
            
            
            #compute gradients of all variables wrt loss
            loss.backward()
            
            #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            #torch.nn.utils.clip_grad_norm(model_obj.model.parameters(), 5.0)
    
            # performs updates using calculated gradients
            model_obj.optimizer.step()
            progress.update(1)
            
    def model_train(self, params, model_obj, tr_proc_data, train_y, features_dict, data_eval):
        """
        Train the neural model until a certain number of epochs.
        :param params:
        :param model_obj:
        :param tr_proc_data:
        :param train_y:
        :param features_dict:
        :param data_eval:
        :return:
        """
        num_steps = int(np.ceil(len(train_y)/params.batch_size))
        eval_epochs = []
        max_F1 = float('-inf')
        for epoch in range(params.num_epochs):
            progress = tqdm.tqdm(total=num_steps, ncols=75, desc='Train epoch {} of {}'.format(epoch+1, params.num_epochs))
            data_iterator = batch.batch_preparation().get_a_batch(model_obj, tr_proc_data, train_y, features_dict, num_steps, params.batch_size, params.device, True)    
            self.train(epoch, model_obj, data_iterator, num_steps, progress, data_eval["val"])
            if data_eval:                
                val_eval_dict,_,_,_ = evaluate.evaluate_model().get_eval(model_obj, data_eval["val"]["val_x"], data_eval["val"]["val_y"], data_eval["val"]["sent_lens"],data_eval["val"]["index2token"], data_eval["val"]["label_tag"], data_eval["val"]["pad_name"])
                eval_epochs.append(val_eval_dict )
                           
                #Save the best model
                if val_eval_dict["F1"] > max_F1:
                    max_F1 = val_eval_dict["F1"]
                    print("\nBest F1: {}. Saving best model....\n".format(val_eval_dict["F1"]))
                    torch.save(model_obj.model.state_dict(), model_obj.best_model_path)
                    patience = 0
                else:
                    patience += 1
                    if patience > model_obj.patience:
                        break
            progress.close()
                                
        return eval_epochs