# -- ------------------------------------------------------------------------------------ -- #
# -- proyecto: Microestructura y Sistemas de Trading - Laboratorio 4 - Analisis Fundamental
# -- archivo: visualizaciones.py - para visualizacion de datos
# -- mantiene: Carlos Nuño, Santiago Barba, Juan Mario
# -- repositorio: https://github.com/CarlosNuno98/LAB_4_CENT
# -- ------------------------------------------------------------------------------------ -- #

import funciones as fn

df_escenarios = fn.df_escenarios()
df_escenarios

df_decisiones = fn.df_decisiones()
df_decisiones['Analisis']
df_decisiones['Decisiones']


df_backtest = fn.df_backtest()
df_backtest

df_backtest_h2 = fn.df_backtest_h()
df_backtest_h2