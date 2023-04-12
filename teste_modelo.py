import pandas as pd
import numpy as np
from sklearn import linear_model
import pickle
#Leitura do dataframe e filtragem do dataframe
df = pd.read_csv('FuelConsumptionCo2.csv')
df = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']]

#Carregar o modelo a partir do arquivo modelo.plk
with open('modelo.pkl', 'rb') as arquivo:
    modelo_carregado = pickle.load(arquivo)

#Usar o modelo carregado para fazer previsões, alterar o nome da coluna e mostrar seus resultados em formato de DataFrame
previsoes = modelo_carregado.predict(df)
Resultado_modelo_previsao = pd.DataFrame(previsoes)
Resultado_modelo_previsao = Resultado_modelo_previsao.rename(columns={Resultado_modelo_previsao.columns.values[0]: "CO2_EMISSAO"})
print(Resultado_modelo_previsao.head(10))