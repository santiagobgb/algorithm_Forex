# -*- coding: utf-8 -*-
"""
Created on Sat May  2 19:02:47 2020

@author: Juanma
"""
#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf #función para graficar autocorrelación
from statsmodels.graphics.tsaplots import plot_pacf #función para graficar autocorrelación parcial
import statsmodels.api as sm
#import statsmodels.stats.diagnostics as smd
import seaborn as sns
sns.set()
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson
from statsmodels.tsa.stattools import adfuller

#%%importamos datos
data_DGO = pd.read_csv("../Proyecto_Equipo_3-master/Indice")

#%%Convertimos a DataFrame
df_DGO = pd.DataFrame(data_DGO)
df_DGO.sort_index(ascending = False, inplace = True)

#%%Gráfica de comportamiento histórico
plt.plot(df_DGO.iloc[-120:-1,0], df_DGO.iloc[-120:-1,1])
plt.show()
#%%Pruebas estad'isticas
#Por el momento utlizaremos solo los íltimos 120 datos del histórico
df_DGO60 = df_DGO.iloc[-120:-1,:]
df_DGO60.describe()
#%%Autocorrelación
plot_acf(df_DGO60.iloc[:,1])
plt.show()

#%%Autocorrelación parcial
plot_pacf(df_DGO60.iloc[:,1])
plt.show()

#%%Regresión lineal
def reglin(x,y):
    x = sm.add_constant(x)
    model = sm.OLS(y,x).fit()
    B0 = model.params[0]
    B1 = model.params[1]
    x = x[:,1]   
    #resumen de la gráfica
    x2 = np.linspace(x.min(), x.max(), 100)
    y_hat = B0 + B1*x2
    plt.scatter(x, y, alpha = 1)
    plt.plot(x2, y_hat, 'r', alpha = 1)
    plt.xlabel('Fecha')
    plt.ylabel('Puntaje Durable Goods Orders')
    return model, B0, B1

#%%
n = 50
x = np.random.randint(0, 100, n)
e = np.random.normal(0, 1, n)

y = 10 + 0.5*x+e

reglin(x,y)
print('La linea de mejor ajuste es: Y={0} + {1}*X'.format(reglin(x,y)[1], reglin(x,y)[2]))

#%%Residuales
modelo, B0, B1 = reglin(x,y)
residuales = modelo.resid
plt.errorbar(x, y, xerr = 0, yerr = [residuales, 0*residuales], linestyle = 'None', color = 'Green');

#%%diagnóstico residuales
plt.scatter(modelo.predict(), residuales)
plt.axhline(0, color = 'red')
plt.xlabel('Valores Predictivos')
plt.ylabel('Residuales')
#plt.xlim[1,50]

#%%Prueba de Heteroscedasticidad
n = 50 
x = np.random.randint(0, 100, n)
e = np.random.normal(0, 1, n)
Y_heteroscedastico = 100 + 2*x + e*x

modelo = sm.OLS(Y_heteroscedastico, sm.add_constant(x)).fit()
B0, B1 = modelo.params
residuales = modelo.resid

plt.scatter(modelo.predict(), residuales)
plt.axhline(0, color = 'red')
plt.xlabel('Valores Predictivos')
plt.ylabel('Residuales');

modelo.summary()

#%%Prueba de hipótesis Breusch-Pagan
#breusch_pagan_p = smd.het_breuschpagan(modelo.resid, modelo.model.exog)[1]
#print(breusch_pagan_p)
#if breusch_pagan_p > 0.05:
#    print('La relación no es heteroscedástica')
#if breusch_pagan_p < 0.05:
#    print('La relación es heteroscedástica')






#Referencias:
#https://medium.com/@calaca89/entendiendo-la-regresi%C3%B3n-lineal-con-python-ed254c14c20

#%%Prueba de normalidad
np.random.seed(1)
datanorm = 5*np.random.randn(100) + 50 #generación de observaciones

print('mean=%.3f stdv=%.3f' % (np.mean(df_DGO60.iloc[:,1]), np.std(df_DGO60.iloc[:,1]))) #resumen

#Pruebas de normalidad visuales
plt.hist(df_DGO60.iloc[:,1], bins = 20) #histograma
plt.show()

#Utilización de QQplot
qqplot(df_DGO60.iloc[:,1], line='s')
plt.show()

#Utilización de Shapiro
stat, p = shapiro(df_DGO60.iloc[:,1])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpretación
alpha = 0.05
if p > alpha:
	print('Sample looks Gaussian (fail to reject H0)')
else:
	print('Sample does not look Gaussian (reject H0)')
    
#D’Agostino’s K^2 Test
stat, p = normaltest(df_DGO60.iloc[:,1])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Sample looks Gaussian (fail to reject H0)')
else:
	print('Sample does not look Gaussian (reject H0)')
    
#Tests – Anderson-Darling Test
result = anderson(df_DGO60.iloc[:,1])
print('Statistic: %.3f' % result.statistic)
p = 0
for i in range(len(result.critical_values)):
	sl, cv = result.significance_level[i], result.critical_values[i]
	if result.statistic < result.critical_values[i]:
		print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
	else:
		print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
        
#https://datascienceplus.com/normality-tests-in-python/
#%%Prueba de estacionariedad:No es estacionaria
one, two, three = np.split(
        df_DGO60.iloc[:,1].sample(
        frac=1), [int(.25*len(df_DGO60.iloc[:,1])),
        int(.75*len(df_DGO60.iloc[:,1]))])
        
mean1, mean2, mean3 = one.mean(), two.mean(), three.mean()
var1, var2, var3 = one.var(), two.var(), three.var()

print(mean1, mean2, mean3)
print(var1, var2, var3)

#%%Prueba de estacionariedad Augmented Dickey-Fuller: No es estacionario
adf_test = adfuller(df_DGO60.iloc[:,1])

print( "ADF = " + str(adf_test[0]))
print( "p-value = " +str(adf_test[1]))



pd.plotting.lag_plot(df_DGO60.iloc[:,1])
#https://pythondata.com/stationary-data-tests-for-time-series-forecasting/
#https://medium.com/datos-y-ciencia/modelos-de-series-de-tiempo-en-python-f861a25b9677 este link nos puede ser útil

#%%Prueba de estacionalidad

#Descomposición de serie de tiempo
descomposicion = sm.tsa.seasonal_decompose(df_DGO60.iloc[:,1],
                                                  model='additive', freq=30)  
fig = descomposicion.plot()
#https://relopezbriega.github.io/blog/2016/09/26/series-de-tiempo-con-python/


#%%

retorno = []

for i in range(1,len(df_DGO)):
    ret = (df_DGO.iloc[i,1]/df_DGO.iloc[i-1,1])
    retorno.append(ret)
    
plt.plot(retorno)
plt.plot(df_DGO['Actual'])
plt.show()

vardf = np.var(df_DGO['Actual'])
varret = np.var(retorno)


one, two, three = np.split(
        df_DGO.iloc[:,1].sample(
        frac=1), [int(.25*len(df_DGO.iloc[:,1])),
        int(.75*len(df_DGO.iloc[:,1]))])
        
mean1, mean2, mean3 = one.mean(), two.mean(), three.mean()
var1, var2, var3 = one.var(), two.var(), three.var()
    
meanvar = {'Medias':[mean1, mean2, mean3], 'Varianzas':[var1, var2, var3]}

df_estacionariedad = pd.DataFrame(meanvar)