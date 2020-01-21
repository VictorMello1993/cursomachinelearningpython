# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 16:36:23 2020

@author: santo
"""

#-----------------------------------Pré-carregamento dos dados----------------------------------------------------------------------------------
import pandas as pd
base = pd.read_csv('credit_data.csv')

#exibindo os valores estatísticos da base de dados do arquivo csv 
#(quantidade, média, desvio padrão, quartis, etc...)
base.describe()

#-----------------------------------Tratamento dos dados inconsistentes---------------------------------------------------------------
#localizando valores inconsistentes - idade negativa
base.loc[base['age'] < 0]

#apagar a coluna idade
base.drop('age', 1, inplace=True)

#apagar somente registros inconsistentes
base.drop(base[base.age < 0].index, inplace=True)

#preencher os valores manualmente
#preencher os valores com a média
base.mean()

#visualizar a média de um atributo, considerando registros inconsistentes - idade
base['age'].mean()

#visualizar a média de um atributo, considerando registros válidos - idade
base['age'][base.age > 0].mean()

#substituindo registros com idade negativa pela média de idade dos registros válidos (SOLUÇÃO) 
base.loc[base.age < 0, 'age'] = 40.92

#------------------------------------------Tratamento dos dados faltantes------------------------------------------------------------

#Verificando todos registros localizando algum valor null (retorno booleano)
pd.isnull(base['age'])

#Obtendo registros com atributo idade com valor null 
base.loc[pd.isnull(base['age'])]

#Obtendo uma divisão da base de dados
#obtendo um DataFrame de atributos previsores
previsores = base.iloc[:, 1:4].values

#obtendo um DataFrame de atributo classe
classe = base.iloc[:, 4].values

from sklearn.preprocessing import Imputer
imputer = Imputer()



