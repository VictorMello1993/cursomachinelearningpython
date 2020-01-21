# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 13:08:42 2020

@author: santo
"""

import pandas as pd
base = pd.read_csv('census.csv')

classe = base.iloc[:,14].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

previsores = base.iloc[:,0:14].values
#previsores = base.iloc[:,8:9].values

labelencoder_previsores = LabelEncoder()

#Convertendo os valores do atributo raça para valores numéricos
#labels = labelencoder_previsores.fit_transform(previsores[:,1])

#Convertendo todos os valores das colunas que retornam string em valores numéricos, pois a maioria
#dos algortimos de machine learning se baseia em cálculos de equações que tratam valores numéricos
previsores[:,1] = labelencoder_previsores.fit_transform(previsores[:,1]) #classe de trabalho
previsores[:,3] = labelencoder_previsores.fit_transform(previsores[:,3]) #escolaridade
previsores[:,5] = labelencoder_previsores.fit_transform(previsores[:,5]) #estado civil
previsores[:,6] = labelencoder_previsores.fit_transform(previsores[:,6]) #cargo (profissão)
previsores[:,7] = labelencoder_previsores.fit_transform(previsores[:,7]) #relacionamento
previsores[:,8] = labelencoder_previsores.fit_transform(previsores[:,8]) #raça
previsores[:,9] = labelencoder_previsores.fit_transform(previsores[:,9]) #sexo
previsores[:,13] = labelencoder_previsores.fit_transform(previsores[:,13]) #país de origem

#convertendo todos os valores das variáveis categóricas em numéricas (0 ou 1) usando one hot encoder
onehotenconder = OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13])
previsores = onehotenconder.fit_transform(previsores).toarray()

labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)


