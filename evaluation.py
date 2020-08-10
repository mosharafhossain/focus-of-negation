# -*- coding: utf-8 -*-
"""
@author: Md Mosharaf Hossain
Doctoral Student in CSE at the University of North Texas
"""


import numpy as np

PAD = "PAD"
FOCUS = ["I_F"]

class evaluation():
    def get_label_3Dto2D(self, test_pred, index2label):
        """
        Convert label 3D to 2D.
        :param test_pred:
        :param index2label:
        :return:
        """
        test_pred = np.argmax(test_pred, axis=-1)  # applying argmax on last dimension. 
        test_pred = [[index2label[i] for i in sent] for sent in test_pred ]
        return test_pred

    def predict_test(self, model, test_x, index2label):
        """
        Predict Test dataset.
        :param model:
        :param test_x:
        :param index2label:
        :return:
        """
        test_pred = model.predict(test_x)   # test_predict 3D array. 1st dim: number of rows, 2nd dim: label type index, 3rd dim: one-hot-vector of that label type
        test_pred = self.get_label_3Dto2D(test_pred, index2label)
        return test_pred
    
    def get_measures(self, model,test_x, test_y, index2label, label_type):
        """
        Calculate evaluation scores.
        :param model:
        :param test_x:
        :param test_y:
        :param index2label:
        :param label_type:
        :return:
        """
        test_pred = self.predict_test(model, test_x, index2label )        
        test_y = self.get_label_3Dto2D(test_y, index2label)
        
        if label_type == "focus": true_labels = FOCUS
    
            
        tp, tn, fp, fn = 0, 0, 0, 0
        for i in range(len(test_y)):
            for j in range(len(test_y[i])):
                if test_y[i][j]!= PAD: 
                    if test_y[i][j] in true_labels:  
                        if test_pred[i][j] == test_y[i][j]:
                            tp = tp+1
                        else:
                            fn = fn+1
                    else:  # No Cue/ Out of scope case 'N_C'
                        if test_pred[i][j] == test_y[i][j]:
                            tn = tn+1
                        else:
                            fp = fp+1
                            
                else:break
            
        
        cm ={"tp":tp, "tn":tn, "fp":fp, "fn":fn}
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1 = 2*precision*recall/(precision+recall)
        measures = {"cm":cm, "precision":precision, "recall":recall, "f1": f1}
        return measures