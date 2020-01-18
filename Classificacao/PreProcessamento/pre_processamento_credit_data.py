# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 16:36:23 2020

@author: santo
"""
import pandas as pd
base = pd.read_csv('credit_data.csv')

#exibindo os valores estatísticos da base de dados do arquivo csv 
#(quantidade, média, desvio padrão, quartis, etc...)
base.describe()

