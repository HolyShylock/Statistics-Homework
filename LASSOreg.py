# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 09:25:20 2021

@author: Ieff Chan
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV

x_data = pd.read_csv('heart_failure_clinical_records_dataset.csv',usecols=list(range(12)))
y_data = pd.read_csv('heart_failure_clinical_records_dataset.csv',usecols=[12])
x_array_data = np.array(x_data)#df数据转为np.ndarray()
y_array_data = np.array(y_data)
x_list_data = x_array_data.tolist()
y_list_data = y_array_data.tolist()
''' 通过交叉检验来获取最优参数'''

lassocv = LassoCV()
lassocv.fit(x_list_data,y_list_data)
alpha = lassocv.alpha_
print('利用Lasso交叉检验计算得出的最优alpha：' + str(alpha))
 
 
'''lasso回归'''
lasso = Lasso(alpha)
lasso.fit(x_list_data,y_list_data)
print(lasso.coef_)
 
'''计算系数不为0的个数'''

n = np.sum(lasso.coef_ != 0)
print('Lasso回归后系数不为0的个数：' + str(n))
 
'''输出结果
   如果names没有定义，则用X1 X2 X3代替
   如果Sort = True，会将系数最大的X放在最前'''
def pretty_print_linear(coefs, names = None, sort = False):  
    if names == None:  
        names = ["X%s" % x for x in range(len(coefs))]  
    lst = zip(coefs, names)  
    if sort:  
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))  
    return " + ".join("%s * %s" % (round(coef, 3), name)  
                                   for coef, name in lst)  
 
 
 
print('Y = '+ pretty_print_linear(lasso.coef_))
