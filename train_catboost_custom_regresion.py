#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  serve to get model prediction 
#  data is in data path
#  model is in model path
#  

#  
#  
#import pickle
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
#import time
import os
import argparse
from six.moves import xrange



class MyLoss_SE():
    # 
    # costom loss function for the regrasion
    #
    def __init__ (self, classes, error=1e-3):
        self.classes = classes
        self.error = error
        #self.weights = weights
        
    def calc_ders_range(self, approxes, targets, weights):
        # approxes is list of floats   
        # targets is float list
        # weights is list of floats
        # weights parameter can be None.
        # Returns first and secornd devirasion 
        
        assert len(approxes) == len(targets)
        n = len(approxes)

        result = []

        for ind in xrange(n):
            approxe = float(approxes[ind])
            target = float(targets[ind]) - self.error
            der1 = approxe-target if (approxe > target) & (self.classes[ind] == 1) else 0
            der1 = approxe-target if (approxe <= target) & (self.classes[ind] == 0) else 0
            der2 = 1

            if weights is not None:
                der1 *= weights[ind]
                der2 *= weights[ind]

            result.append((der1, der2))
            
        return result
        
        
# Train catboost regression model
model = None

def extract_str(text):
    x0 = text.find("(") +1
    x1 = text.find(")")
    if x1 == -1:
        x1 = len(text)
    return text[x0:x1]

def load_data(data_path, dsp_txt='dsp.txt', dsp=3200, f_loss=None):
    
    global df_train, y_train, s_weights, classes 
    
    
    df = pd.read_csv(data_path)
    df.columns = [extract_str(s) for s in df.columns]
    #df['log_BF'] = df['BF'].apply(lambda x: np.log(x) if x>0 else np.log(1e-14))
    #df['log_BP'] = df['BidPrice'].apply(lambda x: np.log(x) if x>0 else np.log(1e-14))
    df['Day'] = pd.to_datetime(df['Date']).dt.day_name() # extract day of week from date
    
    dsp_join = pd.read_csv(dsp_txt, header=None) # list of dsp that are the same as the dsp
    df['Dsp'] = df['Dsp'].apply(lambda x: dsp if x in dsp_join[0].tolist() else x) # replace the dsp name the same
    classes = df['Views'].values
    
    df['Views'] = df['Views'].apply(lambda x: x if x in [0,1] else 0) # views are only {0, 1}
    
    categorical_cols = ['Day', 'Environment', 'DataCenter', 'AdType', 'Browser', 'Hour',
       'DeviceType', 'Platform', 'Dsp', 'Country', 'Size',
       'Profile', 'Zone', 'Maker', 'Model', 'OS', 'ScreenResolution', 'Region',
       'Domain', 'ExternalPublisherId', 'City', 'Carrier']
       
    df_train = df[categorical_cols].fillna(value='Nan').copy() # fill missing values by Nan
    df_train['BF'] = df['BF'].apply(lambda x: x if x >0 else 0) # BF only positive
    #df_train['log_BF'] = df['log_BF']
    
    #y_train = ((df['log_BP'] - df_train['log_BF']) * df['Views']).values # targets
    y_train = df['BidPrice'] - df_train['BF']  # targets
    if not f_loss:
        y_train = y_train * df['Views'].values
    
    not_nan_mask = ~np.isnan(y_train* df['Views'].values)
    
    class_weights = [np.float(df['Views'].value_counts()[1])/df['Views'].value_counts()[0], 1] # weights of classes {0,1}
    s_weights = (df['Views'].apply(lambda x: class_weights[1] if x == 1 else class_weights[0])).values # weights of samples as classes
    
    df_train =  df_train[not_nan_mask]
    y_train = y_train[not_nan_mask]
    s_weights = s_weights[not_nan_mask] 
    classes = classes[not_nan_mask] 
    
    print ('load_data', df.shape)
    
    
    return categorical_cols


def train_model(data_path, **kwarg):
    
    
    iterations=kwarg.get('iterations', 60)
    depth = kwarg.get('depth',3)
    learning_rate = kwarg.get('learning_rate',0.1)
    f_loss = kwarg.get('f_loss', 'RMSE')
    
    verbose = kwarg.get('verbose',True)
    #m_file = os.path.basename(data_path).split('.')[0] + '_' + str(int(time.time())) + '.cbm'
    m_file = 'model.cbm'
    model_path = kwarg.get('model_path', m_file) 
                           
    categorical_cols = load_data(data_path)
    if f_loss == 'custom':
        f_loss = MyLoss_SE(classes)
        print(f_loss)
    # model variable refers to the global variable
    model = CatBoostRegressor(iterations=iterations, 
                              depth=depth,
                              learning_rate=learning_rate,
                              loss_function=f_loss,
                              eval_metric='RMSE',
                              cat_features = categorical_cols,
                              verbose=verbose)
    # CatBoost data pool for training with sample weights                         
    data_tr = Pool(df_train, weight=s_weights, cat_features=categorical_cols, label=y_train)
    
    print('start_train')
    model.fit(data_tr)
    model.save_model(model_path)
    
    
    return model, model_path

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
      '--data',
      type=str,
      default = 'new_data.csv',
      help='Path to data for train as csv')
     
    parser.add_argument(
      '--model',
      type=str,
      default = 'model.cbm',
      help='Path to trained model')
      
    parser.add_argument(
      '--f_loss',
      type=str,
      default = None,
      help='Loss function for optimization')
      
    kwargs, unparsed = parser.parse_known_args()
    
    print('train data',kwargs.data)
    if not os.path.exists(os.path.dirname(kwargs.model)):
        os.makedirs(os.path.dirname(kwargs.model))
    
    m, name_model = train_model(kwargs.data, model_path=kwargs.model, f_loss=kwargs.f_loss)
    
    print('save model in', name_model)  
