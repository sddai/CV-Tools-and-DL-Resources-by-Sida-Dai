# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 15:24:19 2015
 
@author: hd
"""
 
from sklearn import cross_validation
 
c = []
j=0
filename = r'C:\Users\hd\Desktop\bookmarks\bookmarks.arff'
out_train = open(r'C:\Users\hd\Desktop\bookmarks\train.arff','w')
out_test = open(r'C:\Users\hd\Desktop\bookmarks\test.arff','w')
 
for line in open(filename):
#    items = line.strip().split()
    c.append(line)
 
c_train,c_test = cross_validation.train_test_split(c,test_size = 0.6)
for i in c_train:
    out_train.write(i)
for i in c_test:
    out_test.write(i)
