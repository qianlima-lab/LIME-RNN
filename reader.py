# -*- coding: utf-8 -*- 
from __future__ import absolute_import
from __future__ import division

import collections
import os
import numpy as np
import tensorflow as tf
import math

def read_data_asmatrix_minmax(name,dim):
    fileObj = open(name,'r')
    data=fileObj.read().split()
    fileObj.close()
    data = np.array(data, dtype=np.float16).reshape(-1,dim)
    for i in xrange(dim):
        maxx=max(data[:,i])
        minn=min(data[:,i])
        data[:,i]=(data[:,i]-minn)/(maxx-minn)
    return data

def read_data_asmatrix(name,dim):
    fileObj = open(name,'r')
    data=fileObj.read().split()
    fileObj.close()
    data = np.array(data, dtype=np.float16).reshape(-1,dim)
    return data

def count_RMSE_matrix(raw,fi,res,dim,num_steps):
    RMSE=0.0
    count=0
    for i in xrange(len(res)):
        for j in xrange(dim):
            if fi[i+num_steps,j]==-1:
                RMSE+=math.pow(raw[i+num_steps,j]-res[i,j],2)
                count+=1
    return math.sqrt(RMSE*1.0/count)

def ptb_iterator(raw_data, batch_size, num_steps,dim): 
  fileObj = open("data/raw.txt",'r')
  raw=fileObj.read().split()
  fileObj.close()
  raw = np.array(raw, dtype=np.float32).reshape(-1,dim)
  maxx=np.max(raw,axis=0)
  minn=np.min(raw,axis=0)
  raw_data = np.array(raw_data, dtype=np.float32).reshape(-1) 
  data_len = len(raw_data) 	
  row = data_len // dim  
  data_num =row-num_steps+1
  if batch_size>=(data_num-1) or ((data_num-1)%batch_size!=0) :
      raise ValueError("Error, decrease batch_size or num_steps")
  data = np.reshape(raw_data,[-1,dim])
  for i in xrange(dim):
      for j in xrange(row):
          if data[j,i]!=-1:
              data[j,i]=(data[j,i]-minn[i])/(maxx[i]-minn[i])
  data2 = np.zeros([data_num,num_steps,dim])
  for i in range(data_num):	 
      data2[i] = data[i:i+num_steps,:]

  batch_len=(data_num-1)//batch_size
  for i in range(batch_len):
      x = data2[i*batch_size:(i+1)*batch_size,:,:]
      y = data2[i*batch_size+1:(i+1)*batch_size+1,:,:]
      yield (x,y)

