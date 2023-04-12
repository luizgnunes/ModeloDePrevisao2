import pandas as pd
import numpy as np
from sklearn import linear_model
import pickle
#Leitura do dataframe
df = pd.read_csv('FuelConsumptionCo2.csv')

#Criacao do teste e do treino
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

#Treinamento do modelo
regr = linear_model.LinearRegression()
x_train = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
y_train = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x_train, y_train)
print ('Coeficiente: ', regr.coef_)

#Teste do modelo e seus resultados de MSE e R-Squared
y_train= regr.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
x_test = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y_test = np.asanyarray(test[['CO2EMISSIONS']])
print("Soma residual dos quadrados (MSE): %.2f" % np.mean((y_train - y_test) ** 2))
print('Pontuação de variância: %.2f' % regr.score(x_test, y_test))

#Salvar o modelo em um arquivo utilizando o pickle
with open('modelo.pkl', 'wb') as arquivo:
    pickle.dump(regr, arquivo)


