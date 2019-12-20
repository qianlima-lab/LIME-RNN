# -*- coding: utf-8 -*- 
from __future__ import absolute_import
from __future__ import division

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def read_raw_data(name):
    data = pd.read_csv(name,sep='\t',header=None)
    columns_name = list(data.columns)
    ##minmax-score
    norlizer = MinMaxScaler().fit(data)
    data = norlizer.transform(data)
    return data, columns_name, norlizer

def read_missing_data(name,norlizer,dim):
    fileObj = open(name,'r')
    data=fileObj.read().split()
    fileObj.close()
    data = np.array(data, dtype=np.float16).reshape(-1,dim)
    data [data==-1.0] =np.nan
    data = norlizer.transform(data)
    data[np.isnan(data)] = -1.0
    return data
    
def RMSE_Metric(raw, miss_data, pre, missing_flag):
    raw = np.array(raw).copy()
    miss_data = np.array(miss_data).copy()
    pre = np.array(pre).copy()
    assert miss_data.shape == pre.shape
    assert miss_data.shape == raw.shape
    return np.sqrt(np.mean((raw[ (miss_data == missing_flag)] - pre[(miss_data == missing_flag)])**2))
    

def ptb_iterator(raw_data, batch_size, num_steps,dim):
    row = raw_data.shape[0]
    data_num = row-num_steps+1
    if batch_size>=(data_num-1) or ((data_num-1)%batch_size!=0) :
        raise ValueError("Error, decrease batch_size or num_steps")
    data2 = np.zeros([data_num, num_steps ,dim])
    for i in range(data_num):
        data2[i] = raw_data[i:i+num_steps,:]

    batch_len=(data_num-1)//batch_size
    for i in range(batch_len):
        x = data2[i*batch_size:(i+1)*batch_size,:,:]
        y = data2[i*batch_size+1:(i+1)*batch_size+1,:,:]
        yield (x,y)

if __name__ == '__main__':
    data, columns_name, norlizer = read_raw_data('raw.txt')
    data2 = read_missing_data('miss_data.txt', norlizer, 24)