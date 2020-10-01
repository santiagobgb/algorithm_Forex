# -- ------------------------------------------------------------------------------------ -- #
# -- proyecto: Microestructura y Sistemas de Trading - Laboratorio 4 - Analisis Fundamental
# -- archivo: funciones.py - para procesamiento de datos
# -- mantiene: Carlos Nuño, Santiago Barba, Juan Mario
# -- repositorio: https://github.com/CarlosNuno98/LAB_4_CENT
# -- ------------------------------------------------------------------------------------ -- #



#%%
import pandas as pd  
import numpy as np                                     # dataframes y utilidades
from datetime import timedelta                            # diferencia entre datos tipo tiempo
from oandapyV20 import API                                # conexion con broker OANDA
import oandapyV20.endpoints.instruments as instruments    # informacion de precios historicos
import statsmodels.api as sm
#import arch as arch
#from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf #función para graficar autocorrelación
from statsmodels.graphics.tsaplots import plot_pacf #función para graficar autocorrelación 
import matplotlib.pyplot as plt
import statsmodels.stats.diagnostic as stm
import random
from functools import partial
#import visualizaciones as vz
global lista
global Fechas
global df_decisiones
global Decisiones
# -- --------------------------------------------------------- FUNCION: Descargar precios -- #
# -- Descargar precios historicos con OANDA

def f_precios_masivos(p0_fini, p1_ffin, p2_gran, p3_inst, p4_oatk, p5_ginc):
    """
    Parameters
    ----------
    p0_fini
    p1_ffin
    p2_gran
    p3_inst
    p4_oatk
    p5_ginc

    Returns
    -------
    dc_precios

    Debugging
    ---------

    """

    def f_datetime_range_fx(p0_start, p1_end, p2_inc, p3_delta):
        """

        Parameters
        ----------
        p0_start
        p1_end
        p2_inc
        p3_delta

        Returns
        -------
        ls_resultado

        Debugging
        ---------
        """

        ls_result = []
        nxt = p0_start

        while nxt <= p1_end:
            ls_result.append(nxt)
            if p3_delta == 'minutes':
                nxt += timedelta(minutes=p2_inc)
            elif p3_delta == 'hours':
                nxt += timedelta(hours=p2_inc)
            elif p3_delta == 'days':
                nxt += timedelta(days=p2_inc)

        return ls_result

    # inicializar api de OANDA

    api = API(access_token=p4_oatk)

    gn = {'S30': 30, 'S10': 10, 'S5': 5, 'M1': 60, 'M5': 60 * 5, 'M15': 60 * 15,
          'M30': 60 * 30, 'H1': 60 * 60, 'H4': 60 * 60 * 4, 'H8': 60 * 60 * 8,
          'D': 60 * 60 * 24, 'W': 60 * 60 * 24 * 7, 'M': 60 * 60 * 24 * 7 * 4}

    # -- para el caso donde con 1 peticion se cubran las 2 fechas
    if int((p1_ffin - p0_fini).total_seconds() / gn[p2_gran]) < 4999:

        # Fecha inicial y fecha final
        f1 = p0_fini.strftime('%Y-%m-%dT%H:%M:%S')
        f2 = p1_ffin.strftime('%Y-%m-%dT%H:%M:%S')

        # Parametros pra la peticion de precios
        params = {"granularity": p2_gran, "price": "M", "dailyAlignment": 16, "from": f1,
                  "to": f2}

        # Ejecutar la peticion de precios
        a1_req1 = instruments.InstrumentsCandles(instrument=p3_inst, params=params)
        a1_hist = api.request(a1_req1)

        # Para debuging
        # print(f1 + ' y ' + f2)
        lista = list()

        # Acomodar las llaves
        for i in range(len(a1_hist['candles']) - 1):
            lista.append({'TimeStamp': a1_hist['candles'][i]['time'],
                          'Open': a1_hist['candles'][i]['mid']['o'],
                          'High': a1_hist['candles'][i]['mid']['h'],
                          'Low': a1_hist['candles'][i]['mid']['l'],
                          'Close': a1_hist['candles'][i]['mid']['c']})

        # Acomodar en un data frame
        r_df_final = pd.DataFrame(lista)
        r_df_final = r_df_final[['TimeStamp', 'Open', 'High', 'Low', 'Close']]
        r_df_final['TimeStamp'] = pd.to_datetime(r_df_final['TimeStamp'])
        r_df_final['Open'] = pd.to_numeric(r_df_final['Open'], errors='coerce')
        r_df_final['High'] = pd.to_numeric(r_df_final['High'], errors='coerce')
        r_df_final['Low'] = pd.to_numeric(r_df_final['Low'], errors='coerce')
        r_df_final['Close'] = pd.to_numeric(r_df_final['Close'], errors='coerce')

        return r_df_final

    # -- para el caso donde se construyen fechas secuenciales
    else:

        # hacer series de fechas e iteraciones para pedir todos los precios
        fechas = f_datetime_range_fx(p0_start=p0_fini, p1_end=p1_ffin, p2_inc=p5_ginc,
                                     p3_delta='minutes')

        # Lista para ir guardando los data frames
        lista_df = list()

        for n_fecha in range(0, len(fechas) - 1):

            # Fecha inicial y fecha final
            f1 = fechas[n_fecha].strftime('%Y-%m-%dT%H:%M:%S')
            f2 = fechas[n_fecha + 1].strftime('%Y-%m-%dT%H:%M:%S')

            # Parametros pra la peticion de precios
            params = {"granularity": p2_gran, "price": "M", "dailyAlignment": 16, "from": f1,
                      "to": f2}

            # Ejecutar la peticion de precios
            a1_req1 = instruments.InstrumentsCandles(instrument=p3_inst, params=params)
            a1_hist = api.request(a1_req1)

            # Para debuging
            print(f1 + ' y ' + f2)
            lista = list()

            # Acomodar las llaves
            for i in range(len(a1_hist['candles']) - 1):
                lista.append({'TimeStamp': a1_hist['candles'][i]['time'],
                              'Open': a1_hist['candles'][i]['mid']['o'],
                              'High': a1_hist['candles'][i]['mid']['h'],
                              'Low': a1_hist['candles'][i]['mid']['l'],
                              'Close': a1_hist['candles'][i]['mid']['c']})

            # Acomodar en un data frame
            pd_hist = pd.DataFrame(lista)
            pd_hist = pd_hist[['TimeStamp', 'Open', 'High', 'Low', 'Close']]
            pd_hist['TimeStamp'] = pd.to_datetime(pd_hist['TimeStamp'])

            # Ir guardando resultados en una lista
            lista_df.append(pd_hist)

        # Concatenar todas las listas
        r_df_final = pd.concat([lista_df[i] for i in range(0, len(lista_df))])

        # resetear index en dataframe resultante porque guarda los indices del dataframe pasado
        r_df_final = r_df_final.reset_index(drop=True)
        r_df_final['Open'] = pd.to_numeric(r_df_final['Open'], errors='coerce')
        r_df_final['High'] = pd.to_numeric(r_df_final['High'], errors='coerce')
        r_df_final['Low'] = pd.to_numeric(r_df_final['Low'], errors='coerce')
        r_df_final['Close'] = pd.to_numeric(r_df_final['Close'], errors='coerce')

        return r_df_final


def df_escenarios():
    from datetime import datetime,timedelta
    from datetime import datetime
    data = pd.read_csv("../Proyecto_Equipo_3-master/Indice")
    #Tomamos la columna de Datetime
    data_new = data["DateTime"]
    
    #Convertimos la columna en un Datetime
    Nuevo = []
    for i in range(0,len(data_new)):
        datetime_str = data_new[i]
        datetime_object = datetime.strptime(datetime_str, '%m/%d/%Y %H:%M:%S')
        Nuevo.append(datetime_object)

    Inicial = pd.DataFrame(Nuevo)
    Inicial["Inicial"] = Inicial
    Inicial = Inicial.drop([0], axis=1)
    
    #Se le agregan los 30 minutos despues de que salio el indicador
    import datetime
    y = datetime.timedelta(minutes=30)
    Datos_30 = Inicial + y
    Inicial["Datos + 30"] = Datos_30
    
    #Definimos las variables de token, la divisa a utilizar y el rango de tiempo
    OA_Ak = '15f72a29c535bc9eb4d8f9c3267c7a85-e152e24cb184171e7d1b098219e4c303' 
    OA_In = "USD_JPY"  
    OA_Gn = "M1"
    
    #Alguinas fechas no cuentan con los datos por lo que se revisaran cuales son para despues 
    #crear un nuevo DataFrame con todas las fechas que si tienen datos
    global lista
    lista = []
    Fecha_inicio = []
    Fecha_final = []
    for i in range(0,len(Inicial)):
        fini = pd.to_datetime(Inicial["Inicial"][i]).tz_localize('GMT') 
        ffin = pd.to_datetime(Inicial["Datos + 30"][i]).tz_localize('GMT') 
        try: 
            stock = f_precios_masivos(p0_fini=fini, p1_ffin=ffin, p2_gran=OA_Gn,
                                     p3_inst=OA_In, p4_oatk=OA_Ak, p5_ginc=4900)
            Data = pd.DataFrame(stock)
            lista.append(Data)
            Fecha_inicio.append(fini)
            Fecha_final.append(ffin)
        except:
            pass 
        
    global Fechas
    Fechas = pd.DataFrame(Fecha_inicio)
    Fechas["Fecha Inicial"] = Fechas
    Fechas = Fechas.drop([0], axis=1)
    Fechas["Fecha Final"] = pd.DataFrame(Fecha_final)
    
    # Se crea un DataFrame en donde se calculara la direccion(positiva o negativa), los pips alcistas, bajistas y volatilidad
    
    Pips_Direccion = []
    Pips_Alcista = []
    Pips_Bajista = []
    Volatilidad = []

    for i in range(0,len(lista)):

        Data_Direccion = lista[i][["Open", "High", "Low","Close"]]
        Lista = Data_Direccion.values.tolist()

        Open = Lista[0][0]
        High = Data_Direccion["High"].max()
        Low = Data_Direccion["Low"].min()
        Close = Lista[-1][3]


        Direccion = (Close - Open)* 100
        if Direccion > 0:
            Dir = 1
        else: 
            Dir = -1
        Pips_Direccion.append(Dir)

        Alcista = (High - Open) * 100
        Pips_Alcista.append(Alcista)

        Bajista = (Open - Low) * 100
        Pips_Bajista.append(Bajista)

        Vol = (High - Low) * 100
        Volatilidad.append(Vol)

    pips_direccion = pd.DataFrame(Pips_Direccion)
    Fechas["Direccion"] = pips_direccion

    pips_alcistas = pd.DataFrame(Pips_Alcista)
    Fechas["Pips Alcistas"] = pips_alcistas

    pips_bajistas = pd.DataFrame(Pips_Bajista)
    Fechas["Pips Bajistas"] = pips_bajistas

    pips_volatilidad = pd.DataFrame(Volatilidad)
    Fechas["Volatilidad"] = pips_volatilidad
    
    #Determinamos los diferentes escenarios posibles para cada una de las veces que salio el indicador
    
    data_escenarios = data[["DateTime", "Actual", "Consensus", "Previous"]]
    data_escenarios
    
    Escenario = []
    for i in range(0,len(data_escenarios)):
        if data_escenarios["Actual"][i] >= data_escenarios["Consensus"][i] and data_escenarios["Actual"][i] >= data_escenarios["Previous"][i]:
            Escenario.append("A")
        elif data_escenarios["Actual"][i] >= data_escenarios["Consensus"][i] and data_escenarios["Actual"][i] < data_escenarios["Previous"][i]:
            Escenario.append("B")
        elif data_escenarios["Actual"][i] < data_escenarios["Consensus"][i] and data_escenarios["Actual"][i] >= data_escenarios["Previous"][i]:
            Escenario.append("C")
        elif data_escenarios["Actual"][i] < data_escenarios["Consensus"][i] and data_escenarios["Actual"][i] < data_escenarios["Previous"][i]:
            Escenario.append("D")    
    Fechas["Escenarios"] = pd.DataFrame(Escenario)
    
    
    return Fechas

def df_decisiones():
    #Tabla_1 = df_escenarios()
    Entrenamiento = Fechas[["Escenarios", "Direccion", "Pips Alcistas", "Pips Bajistas"]]
    
    Esc_A = Entrenamiento[(Entrenamiento["Escenarios"] == "A")]
    Esc_A_pos = Entrenamiento[(Entrenamiento["Escenarios"] == "A") & (Entrenamiento["Direccion"] == 1)]
    Esc_A_neg = Entrenamiento[(Entrenamiento["Escenarios"] == "A") & (Entrenamiento["Direccion"] == -1)]
    Esc_B = Entrenamiento[(Entrenamiento["Escenarios"] == "B")]
    Esc_B_pos = Entrenamiento[(Entrenamiento["Escenarios"] == "B") & (Entrenamiento["Direccion"] == 1)]
    Esc_B_neg = Entrenamiento[(Entrenamiento["Escenarios"] == "B") & (Entrenamiento["Direccion"] == -1)]
    Esc_C = Entrenamiento[(Entrenamiento["Escenarios"] == "C")]
    Esc_C_pos = Entrenamiento[(Entrenamiento["Escenarios"] == "C") & (Entrenamiento["Direccion"] == 1)]
    Esc_C_neg = Entrenamiento[(Entrenamiento["Escenarios"] == "C") & (Entrenamiento["Direccion"] == -1)]
    Esc_D = Entrenamiento[(Entrenamiento["Escenarios"] == "D")]
    Esc_D_pos = Entrenamiento[(Entrenamiento["Escenarios"] == "D") & (Entrenamiento["Direccion"] == 1)]
    Esc_D_neg = Entrenamiento[(Entrenamiento["Escenarios"] == "D") & (Entrenamiento["Direccion"] == -1)]
    
    Analisis = pd.DataFrame(columns = ["Escenario","Num", "Promedio_Pips_Ganar", "Promedio_Pips_Perder"])
    Analisis.Escenario = ["Escenario A", "A_Postivio", "A_Negativo","Escenario B","B_Postivio", "B_Negativo","Escenario C", "C_Postivio", "C_Negativo", "Escenario D", "D_Postivio", "D_Negativo"]
    Analisis.Num = [len(Esc_A),len(Esc_A_pos),len(Esc_A_neg), len(Esc_B),len(Esc_B_pos),len(Esc_B_neg),len(Esc_C), len(Esc_C_pos),len(Esc_C_neg),len(Esc_D), len(Esc_D_pos),len(Esc_D_neg)]
    Analisis.Promedio_Pips_Ganar = [0,Esc_A_pos["Pips Alcistas"].mean(),Esc_A_neg["Pips Bajistas"].mean(),0,Esc_B_pos["Pips Alcistas"].mean(),Esc_B_neg["Pips Bajistas"].mean(),0,Esc_C_pos["Pips Alcistas"].mean(),Esc_C_neg["Pips Bajistas"].mean(),0,Esc_D_pos["Pips Alcistas"].mean(),Esc_D_neg["Pips Bajistas"].mean()]
    Analisis.Promedio_Pips_Perder = [0,Esc_A_pos["Pips Bajistas"].mean(),Esc_A_neg["Pips Alcistas"].mean(),0,Esc_B_pos["Pips Bajistas"].mean(),Esc_B_neg["Pips Alcistas"].mean(),0,Esc_C_pos["Pips Bajistas"].mean(),Esc_C_neg["Pips Alcistas"].mean(),0,Esc_D_pos["Pips Bajistas"].mean(),Esc_D_neg["Pips Alcistas"].mean()]
    
    #global Decisiones
    Decisiones = pd.DataFrame(columns = ["Escenario", "Operacion", "SL", "TP", "Volumen"])
    Decisiones.Escenario = ["A","B","C","D"]
    Decisiones.Operacion = ["Compra","Compra","Compra", "Venta"]
    Decisiones.SL = [3,4,4,4]
    Decisiones.TP = [7,9,8,8]
    Decisiones.Volumen = [2000,1000,1000,2000]
    
    Tablas = {"Analisis":Analisis,"Decisiones":Decisiones}  
    return Tablas


def df_backtest():
    df_dec = df_decisiones()
    Decis = df_dec['Decisiones']
    
    df_backtest = Fechas[["Fecha Inicial", "Escenarios"]]
    df_backtest

    Operacion = []
    Volumen = []
    pips = []
    resultados = []
    capital = []
    capital_acm = []
    Capital_inicial = 100000

    for i in range(0,len(df_backtest)):  
        #Inicio = lista[i]["Open"][0]
        Pips = 0 
        minuto = 0
        resultado = 0
        #Escenario A

        if df_backtest["Escenarios"][i] == Decis["Escenario"][0]:
            Volumen.append(Decis['Volumen'][0])
            Operacion.append(Decis['Operacion'][0])

            TP = Decis["TP"][0]
            SL = (Decis["SL"][0]) * -1

            while resultado < 1:
                Pips = (lista[i]["Close"][minuto] - lista[i]["Open"][0]) * 100
                minuto = minuto + 1   
                if Pips >= TP:
                    pips.append(TP)
                    resultado = resultado + 1
                elif Pips <= SL:
                    pips.append(SL)
                    resultado = resultado + 1
                elif minuto == len(lista[i]):
                    x = len(lista[i]) -1 
                    Pips = ((lista[i]["Close"][x] - lista[i]["Open"][0]) * 100)
                    pips.append(Pips)
                    resultado = resultado + 1
                else:
                    pass

            if pips[i] > 0:
                resultados.append("ganadora")
            else:
                resultados.append("perdedora")

            Capital = ((Decis['Volumen'][0])/100) * pips[i]
            capital.append(Capital)

        #Escenario B
        elif df_backtest["Escenarios"][i] == Decis["Escenario"][1]:
            Volumen.append(Decis['Volumen'][1])
            Operacion.append(Decis['Operacion'][1])

            TP = Decis["TP"][1]
            SL = (Decis["SL"][1]) * -1

            while resultado < 1:
                Pips = (lista[i]["Close"][minuto] - lista[i]["Open"][0]) * 100
                minuto = minuto + 1   
                if Pips >= TP:
                    pips.append(TP)
                    resultado = resultado + 1
                elif Pips <= SL:
                    pips.append(SL)
                    resultado = resultado + 1
                elif minuto == len(lista[i]):
                    x = len(lista[i]) -1 
                    Pips = ((lista[i]["Close"][x] - lista[i]["Open"][0]) * 100)
                    pips.append(Pips)
                    resultado = resultado + 1
                else:
                    pass

            if pips[i] > 0:
                resultados.append("ganadora")
            else:
                resultados.append("perdedora")

            Capital = ((Decis['Volumen'][0])/100) * pips[i]
            capital.append(Capital)

        #Escenario C
        elif df_backtest["Escenarios"][i] == Decis["Escenario"][2]:
            Volumen.append(Decis['Volumen'][2])
            Operacion.append(Decis['Operacion'][2])

            TP = Decis["TP"][2]
            SL = (Decis["SL"][2]) * -1

            while resultado < 1:
                Pips = (lista[i]["Close"][minuto] - lista[i]["Open"][0]) * 100
                minuto = minuto + 1   
                if Pips >= TP:
                    pips.append(TP)
                    resultado = resultado + 1
                elif Pips <= SL:
                    pips.append(SL)
                    resultado = resultado + 1
                elif minuto == len(lista[i]):
                    x = len(lista[i]) -1 
                    Pips = ((lista[i]["Close"][x] - lista[i]["Open"][0]) * 100)
                    pips.append(Pips)
                    resultado = resultado + 1
                else:
                    pass

            if pips[i] > 0:
                resultados.append("ganadora")
            else:
                resultados.append("perdedora")

            Capital = ((Decis['Volumen'][0])/100) * pips[i]
            capital.append(Capital)

        #Escenario D
        elif df_backtest["Escenarios"][i] == Decis["Escenario"][3]:
            Volumen.append(Decis['Volumen'][3])
            Operacion.append(Decis['Operacion'][3])

            TP = (Decis["TP"][3]) * -1
            SL = (Decis["SL"][3]) 

            while resultado < 1:
                Pips = (lista[i]["Close"][minuto] - lista[i]["Open"][0]) * 100
                minuto = minuto + 1   
                if Pips >= TP:
                    pips.append(TP*-1)
                    resultado = resultado + 1
                elif Pips <= SL:
                    pips.append(SL)
                    resultado = resultado + 1
                elif minuto == len(lista[i]):
                    x = len(lista[i]) -1 
                    Pips = ((lista[i]["Close"][x] - lista[i]["Open"][0]) * 100)
                    pips.append(Pips)
                    resultado = resultado + 1
                else:
                    pass
            if pips[i] > 0:
                resultados.append("ganadora")
            else:
                resultados.append("perdedora")

            Capital = ((Decis['Volumen'][0])/100) * pips[i]
            capital.append(Capital)
        else:
            pass

        Capital_inicial = capital[i] + Capital_inicial
        capital_acm.append(Capital_inicial)


    Operacion = pd.DataFrame(Operacion)
    df_backtest["Operacion"] = Operacion

    Volumen = pd.DataFrame(Volumen)
    df_backtest["Volumen"] = Volumen

    Resultado =  pd.DataFrame(resultados)
    df_backtest["Resultado"] = Resultado

    Pips =  pd.DataFrame(pips)
    df_backtest["Pips"] = Pips

    Capital =  pd.DataFrame(capital)
    df_backtest["Capital"] = Capital

    Capital_acm =  pd.DataFrame(capital_acm)
    df_backtest["Capital_Acm"] = Capital_acm

    return df_backtest



def df_box_jenkins_estacionariedad():

    data_DGO = pd.read_csv("../Proyecto_Equipo_3/Indice")

    df_DGO = pd.DataFrame(data_DGO)
    df_DGO.sort_index(ascending = False, inplace = True)



    df_DGO60 = df_DGO#.iloc[:,1]
    
    one, two, three = np.split(
        df_DGO60.iloc[:,1].sample(
        frac=1), [int(.25*len(df_DGO60.iloc[:,1])),
        int(.75*len(df_DGO60.iloc[:,1]))])
        
    mean1, mean2, mean3 = one.mean(), two.mean(), three.mean()
    var1, var2, var3 = one.var(), two.var(), three.var()
    
    meanvar = {'Medias':[mean1, mean2, mean3], 'Varianzas':[var1, var2, var3]}

    df_estacionariedad = pd.DataFrame(meanvar)

    return df_estacionariedad
    
def df_box_jenkins_estacionalidad():
    
    data_DGO = pd.read_csv("../Proyecto_Equipo_3/Indice")

    df_DGO = pd.DataFrame(data_DGO)
    df_DGO.sort_index(ascending = False, inplace = True)



    df_DGO60 = df_DGO
    
    descomposicion = sm.tsa.seasonal_decompose(df_DGO60.iloc[:,1],
                                                  model='additive', freq=30)  
    fig = descomposicion.plot()
    
    
    return descomposicion



def df_box_jenkins_acf():
    
    data_DGO = pd.read_csv("../Proyecto_Equipo_3/Indice")

    df_DGO = pd.DataFrame(data_DGO)
    df_DGO.sort_index(ascending = False, inplace = True)



    df_DGO60 = df_DGO
    
    plot_acf(df_DGO60.iloc[:,1])
    plt.show()

def df_box_jenkins_pacf():

    data_DGO = pd.read_csv("../Proyecto_Equipo_3/Indice")

    df_DGO = pd.DataFrame(data_DGO)
    df_DGO.sort_index(ascending = False, inplace = True)



    df_DGO60 = df_DGO
    
    plot_pacf(df_DGO60.iloc[:,1])
    plt.show()
    
def df_boxplot():
    

    
    
    data_DGO = pd.read_csv("../Proyecto_Equipo_3-master/Indice")
    
    df_DGO = pd.DataFrame(data_DGO)
    df_DGO.sort_index(ascending = False, inplace = True)
    
    vatip = []
    
    df_DGO60 = df_DGO
    dummy_data = df_DGO60.iloc[:,1]
    
    def make_labels(ax, boxplot):
    
        # Grab the relevant Line2D instances from the boxplot dictionary
        iqr = boxplot['boxes'][0]
        caps = boxplot['caps']
        med = boxplot['medians'][0]
        fly = boxplot['fliers'][0]
        
        # The x position of the median line
        xpos = med.get_xdata()
    
        # Lets make the text have a horizontal offset which is some 
        # fraction of the width of the box
        xoff = 0.10 * (xpos[1] - xpos[0])
    
        # The x position of the labels
        xlabel = xpos[1] + xoff
    
        # The median is the y-position of the median line
        median = med.get_ydata()[1]
    
        # The 25th and 75th percentiles are found from the
        # top and bottom (max and min) of the box
        pc25 = iqr.get_ydata().min()
        pc75 = iqr.get_ydata().max()
    
        # The caps give the vertical position of the ends of the whiskers
        capbottom = caps[0].get_ydata()[0]
        captop = caps[1].get_ydata()[0]
    
        # Make some labels on the figure using the values derived above
        ax.text(xlabel, median,
                'Median = {:6.3g}'.format(median), va='center')
        ax.text(xlabel, pc25,
                '25th percentile = {:6.3g}'.format(pc25), va='center')
        ax.text(xlabel, pc75,
                '75th percentile = {:6.3g}'.format(pc75), va='center')
        ax.text(xlabel, capbottom,
                'Bottom cap = {:6.3g}'.format(capbottom), va='center')
        ax.text(xlabel, captop,
                'Top cap = {:6.3g}'.format(captop), va='center')
    
        # Many fliers, so we loop over them and create a label for each one
        for flier in fly.get_ydata():
            ax.text(1 + xoff, flier,
                    'Flier = {:6.3g}'.format(flier), va='center')
            vatip.append(flier)
    
    # Make the figure
    red_diamond = dict(markerfacecolor='r', marker='D')
    fig3, ax3 = plt.subplots(figsize=(10, 15))
    ax3.set_title('Changed Outlier Symbols')
    
    # Create the boxplot and store the resulting python dictionary
    my_boxes = ax3.boxplot(dummy_data, flierprops=red_diamond)
    
    # Call the function to make labels
    make_labels(ax3, my_boxes)
    
    plt.show();
    
    atip = []
    
    for i in range(0, len(data_DGO)):
        for j in range(0, len(vatip)):
            if data_DGO.iloc[i,1] == vatip[j]:
                atip.append(data_DGO.iloc[i])
      
    df_atip = pd.DataFrame(atip)
    plt.plot(data_DGO.iloc[:,0], data_DGO.iloc[:,1])
    plt.axvline(df_atip.iloc[0,0], color = 'r', linestyle = '--', label='Valores atípicos')
    plt.axvline(df_atip.iloc[1,0], color = 'r', linestyle = '--')
    plt.axvline(df_atip.iloc[2,0], color = 'r', linestyle = '--')
    plt.axvline(df_atip.iloc[3,0], color = 'r', linestyle = '--')
    plt.axvline(df_atip.iloc[4,0], color = 'r', linestyle = '--')
    plt.axvline(df_atip.iloc[5,0], color = 'r', linestyle = '--')
    plt.axvline(df_atip.iloc[6,0], color = 'r', linestyle = '--')
    plt.legend()
    plt.show();
    print(df_atip.iloc[:,0:2])
    return df_atip




def df_parch():
    
    data_DGO = pd.read_csv("../Proyecto_Equipo_3/Indice")
    
    df_DGO = pd.DataFrame(data_DGO)
    df_DGO.sort_index(ascending = False, inplace = True)
    
    
    
    df_DGO60 = df_DGO['Actual']
    
    arc = stm.het_arch(df_DGO60)
    prov = [[0],[0]]
    df_arch = pd.DataFrame(prov, columns = ['Valor'])
    
    df_chivo = pd.DataFrame(arc)
    
    info = np.array([['Prueba de Lagrange (Score Test)'],['p-value']])
    df_arch['Descripción'] = info
    
    reorden = ['Descripción', 'Valor']
    
    
    df_arch = df_arch.reindex(columns = reorden)
    df_chivo = np.reshape(df_chivo, (4,1))
    df_arch.iloc[0,1] = df_chivo.iloc[0,0]
    df_arch.iloc[1,1] = df_chivo.iloc[1,0]
    

    
    return df_arch

def df_backtest_h():
    df_dec = df_decisiones()
    Decis = df_dec['Decisiones']
    
    df_backtest_2 = Fechas[["Fecha Inicial", "Escenarios"]]
    df_backtest_2
    volumen = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    Operacion = []
    Volumen = []
    pips = []
    resultados = []
    capital = []
    capital_acm = []
    TP_aleatorio = []
    SL_aleatorio = []
    Capital_inicial = 100000

    for i in range(0,len(df_backtest_2)):  
        #Inicio = lista[i]["Open"][0]
        Pips = 0 
        minuto = 0
        resultado = 0
        #Escenario A

        if df_backtest_2["Escenarios"][i] == Decis["Escenario"][0]:
            Vol = (random.choice(volumen))
            Volumen.append(Vol)
            Operacion.append(Decis['Operacion'][0])

            TP = (random.randint(1,8))
            TP_aleatorio.append(TP)
            SL = (random.randint(1,5)) * -1
            SL_aleatorio.append(SL)

            while resultado < 1:
                Pips = (lista[i]["Close"][minuto] - lista[i]["Open"][0]) * 100
                minuto = minuto + 1   
                if Pips >= TP:
                    pips.append(TP)
                    resultado = resultado + 1
                elif Pips <= SL:
                    pips.append(SL)
                    resultado = resultado + 1
                elif minuto == len(lista[i]):
                    x = len(lista[i]) -1 
                    Pips = ((lista[i]["Close"][x] - lista[i]["Open"][0]) * 100)
                    pips.append(Pips)
                    resultado = resultado + 1
                else:
                    pass

            if pips[i] > 0:
                resultados.append("ganadora")
            else:
                resultados.append("perdedora")

            Capital = ((Vol)/100) * pips[i]
            capital.append(Capital)

        #Escenario B
        elif df_backtest_2["Escenarios"][i] == Decis["Escenario"][1]:
            Vol = (random.choice(volumen))
            Volumen.append(Vol)
            Operacion.append(Decis['Operacion'][1])

            TP = (random.randint(8,10))
            TP_aleatorio.append(TP)
            SL = (random.randint(3,5)) * -1
            SL_aleatorio.append(SL)

            while resultado < 1:
                Pips = (lista[i]["Close"][minuto] - lista[i]["Open"][0]) * 100
                minuto = minuto + 1   
                if Pips >= TP:
                    pips.append(TP)
                    resultado = resultado + 1
                elif Pips <= SL:
                    pips.append(SL)
                    resultado = resultado + 1
                elif minuto == len(lista[i]):
                    x = len(lista[i]) -1 
                    Pips = ((lista[i]["Close"][x] - lista[i]["Open"][0]) * 100)
                    pips.append(Pips)
                    resultado = resultado + 1
                else:
                    pass

            if pips[i] > 0:
                resultados.append("ganadora")
            else:
                resultados.append("perdedora")

            Capital = ((Vol)/100) * pips[i]
            capital.append(Capital)

        #Escenario C
        elif df_backtest_2["Escenarios"][i] == Decis["Escenario"][2]:
            Vol = (random.choice(volumen))
            Volumen.append(Vol)
            Operacion.append(Decis['Operacion'][2])

            TP = (random.randint(7,9))
            TP_aleatorio.append(TP)
            SL = (random.randint(3,5)) * -1
            SL_aleatorio.append(SL)

            while resultado < 1:
                Pips = (lista[i]["Close"][minuto] - lista[i]["Open"][0]) * 100
                minuto = minuto + 1   
                if Pips >= TP:
                    pips.append(TP)
                    resultado = resultado + 1
                elif Pips <= SL:
                    pips.append(SL)
                    resultado = resultado + 1
                elif minuto == len(lista[i]):
                    x = len(lista[i]) -1 
                    Pips = ((lista[i]["Close"][x] - lista[i]["Open"][0]) * 100)
                    pips.append(Pips)
                    resultado = resultado + 1
                else:
                    pass

            if pips[i] > 0:
                resultados.append("ganadora")
            else:
                resultados.append("perdedora")

            Capital = ((Vol)/100) * pips[i]
            capital.append(Capital)

        #Escenario D
        elif df_backtest_2["Escenarios"][i] == Decis["Escenario"][3]:
            Vol = (random.choice(volumen))
            Volumen.append(Vol)
            Operacion.append(Decis['Operacion'][3])

            TP = (random.randint(7,9)) * -1
            TP_aleatorio.append(TP)
            SL = (random.randint(3,5))
            SL_aleatorio.append(SL)

            while resultado < 1:
                Pips = (lista[i]["Close"][minuto] - lista[i]["Open"][0]) * 100
                minuto = minuto + 1   
                if Pips >= TP:
                    pips.append(TP*-1)
                    resultado = resultado + 1
                elif Pips <= SL:
                    pips.append(SL)
                    resultado = resultado + 1
                elif minuto == len(lista[i]):
                    x = len(lista[i]) -1 
                    Pips = ((lista[i]["Close"][x] - lista[i]["Open"][0]) * 100)
                    pips.append(Pips)
                    resultado = resultado + 1
                else:
                    pass
            if pips[i] > 0:
                resultados.append("ganadora")
            else:
                resultados.append("perdedora")

            Capital = ((Vol)/100) * pips[i]
            capital.append(Capital)
        else:
            pass

        Capital_inicial = capital[i] + Capital_inicial
        capital_acm.append(Capital_inicial)


    Operacion = pd.DataFrame(Operacion)
    df_backtest_2["Operacion"] = Operacion
    
    TP_aleatorio = pd.DataFrame(TP_aleatorio)
    df_backtest_2["TP Aleatorio"] = TP_aleatorio
    
    SL_aleatorio = pd.DataFrame(SL_aleatorio)
    df_backtest_2["SL aleatorio"] = SL_aleatorio

    Volumen = pd.DataFrame(Volumen)
    df_backtest_2["Volumen"] = Volumen

    Resultado =  pd.DataFrame(resultados)
    df_backtest_2["Resultado"] = Resultado

    Pips =  pd.DataFrame(pips)
    df_backtest_2["Pips"] = Pips

    Capital =  pd.DataFrame(capital)
    df_backtest_2["Capital"] = Capital

    Capital_acm =  pd.DataFrame(capital_acm)
    df_backtest_2["Capital_Acm"] = Capital_acm

    return df_backtest_2


def Aopt(Ausar):
    
    def _obj_wrapper(func, args, kwargs, x):
        return func(x, *args, **kwargs)
    
    def _is_feasible_wrapper(func, x):
        return np.all(func(x)>=0)
    
    def _cons_none_wrapper(x):
        return np.array([0])
    
    def _cons_ieqcons_wrapper(ieqcons, args, kwargs, x):
        return np.array([y(x, *args, **kwargs) for y in ieqcons])
    
    def _cons_f_ieqcons_wrapper(f_ieqcons, args, kwargs, x):
        return np.array(f_ieqcons(x, *args, **kwargs))
        
    def pso(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={}, 
            swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100, 
            minstep=1e-8, minfunc=1e-8, debug=False, processes=1,
            particle_output=False):
        """
        Perform a particle swarm optimization (PSO)
       
        Parameters
        ==========
        func : function
            The function to be minimized
        lb : array
            The lower bounds of the design variable(s)
        ub : array
            The upper bounds of the design variable(s)
       
        Optional
        ========
        ieqcons : list
            A list of functions of length n such that ieqcons[j](x,*args) >= 0.0 in 
            a successfully optimized problem (Default: [])
        f_ieqcons : function
            Returns a 1-D array in which each element must be greater or equal 
            to 0.0 in a successfully optimized problem. If f_ieqcons is specified, 
            ieqcons is ignored (Default: None)
        args : tuple
            Additional arguments passed to objective and constraint functions
            (Default: empty tuple)
        kwargs : dict
            Additional keyword arguments passed to objective and constraint 
            functions (Default: empty dict)
        swarmsize : int
            The number of particles in the swarm (Default: 100)
        omega : scalar
            Particle velocity scaling factor (Default: 0.5)
        phip : scalar
            Scaling factor to search away from the particle's best known position
            (Default: 0.5)
        phig : scalar
            Scaling factor to search away from the swarm's best known position
            (Default: 0.5)
        maxiter : int
            The maximum number of iterations for the swarm to search (Default: 100)
        minstep : scalar
            The minimum stepsize of swarm's best position before the search
            terminates (Default: 1e-8)
        minfunc : scalar
            The minimum change of swarm's best objective value before the search
            terminates (Default: 1e-8)
        debug : boolean
            If True, progress statements will be displayed every iteration
            (Default: False)
        processes : int
            The number of processes to use to evaluate objective function and 
            constraints (default: 1)
        particle_output : boolean
            Whether to include the best per-particle position and the objective
            values at those.
       
        Returns
        =======
        g : array
            The swarm's best known position (optimal design)
        f : scalar
            The objective value at ``g``
        p : array
            The best known position per particle
        pf: arrray
            The objective values at each position in p
       
        """
       
        assert len(lb)==len(ub), 'Lower- and upper-bounds must be the same length'
        assert hasattr(func, '__call__'), 'Invalid function handle'
        lb = np.array(lb)
        ub = np.array(ub)
        assert np.all(ub>lb), 'All upper-bound values must be greater than lower-bound values'
       
        vhigh = np.abs(ub - lb)
        vlow = -vhigh
    
        # Initialize objective function
        obj = partial(_obj_wrapper, func, args, kwargs)
        Appenderint = [] #appender
        Appenderg = []
        Appenderfg = []
        # Check for constraint function(s) #########################################
        if f_ieqcons is None:
            if not len(ieqcons):
                if debug:
                    print('No constraints given.')
                cons = _cons_none_wrapper
            else:
                if debug:
                    print('Converting ieqcons to a single constraint function')
                cons = partial(_cons_ieqcons_wrapper, ieqcons, args, kwargs)
        else:
            if debug:
                print('Single constraint function given in f_ieqcons')
            cons = partial(_cons_f_ieqcons_wrapper, f_ieqcons, args, kwargs)
        is_feasible = partial(_is_feasible_wrapper, cons)
    
        # Initialize the multiprocessing module if necessary
        if processes > 1:
            import multiprocessing
            mp_pool = multiprocessing.Pool(processes)
            
        # Initialize the particle swarm ############################################
        S = swarmsize
        D = len(lb)  # the number of dimensions each particle has
        x = np.random.rand(S, D)  # particle positions
        v = np.zeros_like(x)  # particle velocities
        p = np.zeros_like(x)  # best particle positions
        fx = np.zeros(S)  # current particle function values
        fs = np.zeros(S, dtype=bool)  # feasibility of each particle
        fp = np.ones(S)*np.inf  # best particle function values
        g = []  # best swarm position
        fg = np.inf  # best swarm position starting value
        
        # Initialize the particle's position
        x = lb + x*(ub - lb)
    
        # Calculate objective and constraints for each particle
        if processes > 1:
            fx = np.array(mp_pool.map(obj, x))
            fs = np.array(mp_pool.map(is_feasible, x))
        else:
            for i in range(S):
                fx[i] = obj(x[i, :])
                fs[i] = is_feasible(x[i, :])
           
        # Store particle's best position (if constraints are satisfied)
        i_update = np.logical_and((fx < fp), fs)
        p[i_update, :] = x[i_update, :].copy()
        fp[i_update] = fx[i_update]
    
        # Update swarm's best position
        i_min = np.argmin(fp)
        if fp[i_min] < fg:
            fg = fp[i_min]
            g = p[i_min, :].copy()
        else:
            # At the start, there may not be any feasible starting point, so just
            # give it a temporary "best" point since it's likely to change
            g = x[0, :].copy()
           
        # Initialize the particle's velocity
        v = vlow + np.random.rand(S, D)*(vhigh - vlow)
           
        # Iterate until termination criterion met ##################################
        it = 1
        while it <= maxiter:
            rp = np.random.uniform(size=(S, D))
            rg = np.random.uniform(size=(S, D))
    
            # Update the particles velocities
            v = omega*v + phip*rp*(p - x) + phig*rg*(g - x)
            # Update the particles' positions
            x = x + v
            # Correct for bound violations
            maskl = x < lb
            masku = x > ub
            x = x*(~np.logical_or(maskl, masku)) + lb*maskl + ub*masku
    
            # Update objectives and constraints
            if processes > 1:
                fx = np.array(mp_pool.map(obj, x))
                fs = np.array(mp_pool.map(is_feasible, x))
            else:
                for i in range(S):
                    fx[i] = obj(x[i, :])
                    fs[i] = is_feasible(x[i, :])
    
            # Store particle's best position (if constraints are satisfied)
            i_update = np.logical_and((fx < fp), fs)
            p[i_update, :] = x[i_update, :].copy()
            fp[i_update] = fx[i_update]
    
            # Compare swarm's best position with global best position
            i_min = np.argmin(fp)
            if fp[i_min] < fg:
                if debug:
                    print('New best for swarm at iteration {:}: {:} {:}'\
                        .format(it, p[i_min, :], fp[i_min]))
    
                p_min = p[i_min, :].copy()
                stepsize = np.sqrt(np.sum((g - p_min)**2))
    
                if np.abs(fg - fp[i_min]) <= minfunc:
                    print('Stopping search: Swarm best objective change less than {:}'\
                        .format(minfunc))
                    if particle_output:
                        return p_min, fp[i_min], p, fp
                    else:
                        return p_min, fp[i_min]
                elif stepsize <= minstep:
                    print('Stopping search: Swarm best position change less than {:}'\
                        .format(minstep))
                    if particle_output:
                        return p_min, fp[i_min], p, fp
                    else:
                        return p_min, fp[i_min]
                else:
                    g = p_min.copy()
                    fg = fp[i_min]
    
            if debug:
                print('Best after iteration {:}: {:} {:}'.format(it, g, fg))
            it += 1
            Appenderint.append(it)
            Appenderg.append(g)
            Appenderfg.append(fg)
    
        print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))
        
        if not is_feasible(g):
            print("However, the optimization couldn't find a feasible design. Sorry")
        if particle_output:
            return g, fg, p, fp,Appenderint, Appenderg, Appenderfg
        else:
            return g, fg, Appenderint, Appenderg, Appenderfg
        
        
        
        
        
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    import seaborn as sns
    sns.set()

    from sklearn.model_selection import train_test_split
    
    
    
    
    
    
    
    
    
    
    #import pyswarms as pso
    #from pyswarms.utils.functions import single_obj as fx
    #from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
    
    #df_backtest_2 = df_backtest_h#vz.df_backtest_h2
    #df_backtest_2
    
    df_backtest_2 = Ausar
    
    param_A = df_backtest_2[df_backtest_2['Escenarios'] == 'A']#.iloc[5:-1,:]
    param_A = param_A.drop(columns = ['Fecha Inicial', 'Escenarios', 'Operacion','Resultado','Pips','Capital_Acm'])
    column_names = ['Volumen', 'SL aleatorio', 'TP Aleatorio', 'Capital']
    param_A = param_A.reindex(columns = column_names)
    
    targets = param_A['Capital']
    inputs = param_A.drop(['Capital'], axis = 1)
    
    
    x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.1, random_state=365)
    
    # Create a linear regression object
    reg = LinearRegression()
    # Fit the regression with the scaled TRAIN inputs and targets
    reg.fit(x_train,y_train)
    
    
    
    # Obtain the bias (intercept) of the regression
    reg.intercept_
    bint = reg.intercept_
    # Obtain the weights (coefficients) of the regression
    reg.coef_
    # Note that they are barely interpretable if at all
    
    # Create a regression summary where we can compare them with one-another
    reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
    reg_summary['Weights'] = reg.coef_
    reg_summary
        
    
    
    def maxiA(x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        return    reg_summary['Weights'][0]*x1 - reg_summary['Weights'][1]*x2 + reg_summary['Weights'][2]*x3 
        #return bint + -0.00798329*x1 + 41.7485*x2 + 8.16298*x3
    Alb = [1000, 6, 2]
    Aub = [3000, 8, 4]
    
    Axopt, Afopt, iters, xi, fmaxi = pso(maxiA, Alb, Aub) #Axopt : variables óptimas , Afopt: valor máximo, xi: variables ótimas a traves de la iteraciones
    
    Afopt = Afopt+bint
    fmaxi = fmaxi+(bint)
    
    plt.plot(np.arange(1,101,1), fmaxi)
    plt.ylabel('Capital')
    plt.xlabel('Numero de iteraciones')
    plt.title('Optimización contra iteraciones de capital')
    plt.show()
    
    return Axopt, Afopt, bint, reg_summary


def df_histdesv():
    data_DGO = pd.read_csv("../Proyecto_Equipo_3-master/Indice")

    df_DGO = pd.DataFrame(data_DGO)
    df_DGO.sort_index(ascending = False, inplace = True)
    
    df_DGO60 = df_DGO['Actual']
    
    spec = np.std(df_DGO60)
    plt.plot(df_DGO60)
    plt.axhline(spec, color = 'r', label = 'Desviación Estandar: {}'.format(spec))
    plt.axhline(spec*-1,color = 'r')
    plt.legend()
    plt.title('Histórico desde enero 2008')
    plt.ylabel('Resultado del DGO')
    plt.xlabel('Periodo')
    plt.show()
    
    
def df_season():
    data_DGO = pd.read_csv("../Proyecto_Equipo_3-master/Indice")

    df_DGO = pd.DataFrame(data_DGO)
    df_DGO.sort_index(ascending = False, inplace = True)
    
    df_DGO60 = df_DGO['Actual']
    esta = df_DGO.iloc[11:-1,0:2]
    
    esta = esta.drop([10,110], axis = 0)
    indxs = ['Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre']
    p_08 = esta.iloc[0:12,1]
    p_08.index = indxs 
    p_09 = esta.iloc[12:24,1]
    p_09.index = indxs
    p_10 = esta.iloc[24:36,1]
    p_10.index = indxs
    p_11 = esta.iloc[36:48,1]
    p_11.index = indxs
    p_12 = esta.iloc[48:60,1]
    p_12.index = indxs
    p_13 = esta.iloc[60:72,1]
    p_13.index = indxs
    p_14 = esta.iloc[72:84,1]
    p_14.index = indxs
    p_15 = esta.iloc[84:96,1]
    p_15.index = indxs
    p_16 = esta.iloc[96:108,1]
    p_16.index = indxs
    p_17 = esta.iloc[108:120,1]
    p_17.index = indxs
    p_18 = esta.iloc[120:132,1]
    p_18.index = indxs
    p_19 =esta.iloc[132:144,1]
    p_19.index = indxs
    
    
    
    p_08.plot(label = '2008', figsize=(10, 10), linestyle = '--')
    p_09.plot(label = '2009', linestyle = '--')
    p_10.plot(label = '2010')
    p_11.plot(label = '2011')
    p_12.plot(label = '2012')
    p_13.plot(label = '2013')
    p_14.plot(label = '2014')
    p_15.plot(label = '2015')
    p_16.plot(label = '2016')
    p_17.plot(label = '2017')
    p_18.plot(label = '2018', linestyle = ':', color = 'Black')
    p_19.plot(label = '2019', linestyle = ':', color = 'Blue')
    
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=3, fancybox=True, shadow=True)
    plt.show()
    
    ############## Intento b
    
def Bopt(Ausar):
    
    def _obj_wrapper(func, args, kwargs, x):
        return func(x, *args, **kwargs)
    
    def _is_feasible_wrapper(func, x):
        return np.all(func(x)>=0)
    
    def _cons_none_wrapper(x):
        return np.array([0])
    
    def _cons_ieqcons_wrapper(ieqcons, args, kwargs, x):
        return np.array([y(x, *args, **kwargs) for y in ieqcons])
    
    def _cons_f_ieqcons_wrapper(f_ieqcons, args, kwargs, x):
        return np.array(f_ieqcons(x, *args, **kwargs))
        
    def pso(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={}, 
            swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100, 
            minstep=1e-8, minfunc=1e-8, debug=False, processes=1,
            particle_output=False):
        """
        Perform a particle swarm optimization (PSO)
       
        Parameters
        ==========
        func : function
            The function to be minimized
        lb : array
            The lower bounds of the design variable(s)
        ub : array
            The upper bounds of the design variable(s)
       
        Optional
        ========
        ieqcons : list
            A list of functions of length n such that ieqcons[j](x,*args) >= 0.0 in 
            a successfully optimized problem (Default: [])
        f_ieqcons : function
            Returns a 1-D array in which each element must be greater or equal 
            to 0.0 in a successfully optimized problem. If f_ieqcons is specified, 
            ieqcons is ignored (Default: None)
        args : tuple
            Additional arguments passed to objective and constraint functions
            (Default: empty tuple)
        kwargs : dict
            Additional keyword arguments passed to objective and constraint 
            functions (Default: empty dict)
        swarmsize : int
            The number of particles in the swarm (Default: 100)
        omega : scalar
            Particle velocity scaling factor (Default: 0.5)
        phip : scalar
            Scaling factor to search away from the particle's best known position
            (Default: 0.5)
        phig : scalar
            Scaling factor to search away from the swarm's best known position
            (Default: 0.5)
        maxiter : int
            The maximum number of iterations for the swarm to search (Default: 100)
        minstep : scalar
            The minimum stepsize of swarm's best position before the search
            terminates (Default: 1e-8)
        minfunc : scalar
            The minimum change of swarm's best objective value before the search
            terminates (Default: 1e-8)
        debug : boolean
            If True, progress statements will be displayed every iteration
            (Default: False)
        processes : int
            The number of processes to use to evaluate objective function and 
            constraints (default: 1)
        particle_output : boolean
            Whether to include the best per-particle position and the objective
            values at those.
       
        Returns
        =======
        g : array
            The swarm's best known position (optimal design)
        f : scalar
            The objective value at ``g``
        p : array
            The best known position per particle
        pf: arrray
            The objective values at each position in p
       
        """
       
        assert len(lb)==len(ub), 'Lower- and upper-bounds must be the same length'
        assert hasattr(func, '__call__'), 'Invalid function handle'
        lb = np.array(lb)
        ub = np.array(ub)
        assert np.all(ub>lb), 'All upper-bound values must be greater than lower-bound values'
       
        vhigh = np.abs(ub - lb)
        vlow = -vhigh
    
        # Initialize objective function
        obj = partial(_obj_wrapper, func, args, kwargs)
        Appenderint = [] #appender
        Appenderg = []
        Appenderfg = []
        # Check for constraint function(s) #########################################
        if f_ieqcons is None:
            if not len(ieqcons):
                if debug:
                    print('No constraints given.')
                cons = _cons_none_wrapper
            else:
                if debug:
                    print('Converting ieqcons to a single constraint function')
                cons = partial(_cons_ieqcons_wrapper, ieqcons, args, kwargs)
        else:
            if debug:
                print('Single constraint function given in f_ieqcons')
            cons = partial(_cons_f_ieqcons_wrapper, f_ieqcons, args, kwargs)
        is_feasible = partial(_is_feasible_wrapper, cons)
    
        # Initialize the multiprocessing module if necessary
        if processes > 1:
            import multiprocessing
            mp_pool = multiprocessing.Pool(processes)
            
        # Initialize the particle swarm ############################################
        S = swarmsize
        D = len(lb)  # the number of dimensions each particle has
        x = np.random.rand(S, D)  # particle positions
        v = np.zeros_like(x)  # particle velocities
        p = np.zeros_like(x)  # best particle positions
        fx = np.zeros(S)  # current particle function values
        fs = np.zeros(S, dtype=bool)  # feasibility of each particle
        fp = np.ones(S)*np.inf  # best particle function values
        g = []  # best swarm position
        fg = np.inf  # best swarm position starting value
        
        # Initialize the particle's position
        x = lb + x*(ub - lb)
    
        # Calculate objective and constraints for each particle
        if processes > 1:
            fx = np.array(mp_pool.map(obj, x))
            fs = np.array(mp_pool.map(is_feasible, x))
        else:
            for i in range(S):
                fx[i] = obj(x[i, :])
                fs[i] = is_feasible(x[i, :])
           
        # Store particle's best position (if constraints are satisfied)
        i_update = np.logical_and((fx < fp), fs)
        p[i_update, :] = x[i_update, :].copy()
        fp[i_update] = fx[i_update]
    
        # Update swarm's best position
        i_min = np.argmin(fp)
        if fp[i_min] < fg:
            fg = fp[i_min]
            g = p[i_min, :].copy()
        else:
            # At the start, there may not be any feasible starting point, so just
            # give it a temporary "best" point since it's likely to change
            g = x[0, :].copy()
           
        # Initialize the particle's velocity
        v = vlow + np.random.rand(S, D)*(vhigh - vlow)
           
        # Iterate until termination criterion met ##################################
        it = 1
        while it <= maxiter:
            rp = np.random.uniform(size=(S, D))
            rg = np.random.uniform(size=(S, D))
    
            # Update the particles velocities
            v = omega*v + phip*rp*(p - x) + phig*rg*(g - x)
            # Update the particles' positions
            x = x + v
            # Correct for bound violations
            maskl = x < lb
            masku = x > ub
            x = x*(~np.logical_or(maskl, masku)) + lb*maskl + ub*masku
    
            # Update objectives and constraints
            if processes > 1:
                fx = np.array(mp_pool.map(obj, x))
                fs = np.array(mp_pool.map(is_feasible, x))
            else:
                for i in range(S):
                    fx[i] = obj(x[i, :])
                    fs[i] = is_feasible(x[i, :])
    
            # Store particle's best position (if constraints are satisfied)
            i_update = np.logical_and((fx < fp), fs)
            p[i_update, :] = x[i_update, :].copy()
            fp[i_update] = fx[i_update]
    
            # Compare swarm's best position with global best position
            i_min = np.argmin(fp)
            if fp[i_min] < fg:
                if debug:
                    print('New best for swarm at iteration {:}: {:} {:}'\
                        .format(it, p[i_min, :], fp[i_min]))
    
                p_min = p[i_min, :].copy()
                stepsize = np.sqrt(np.sum((g - p_min)**2))
    
                if np.abs(fg - fp[i_min]) <= minfunc:
                    print('Stopping search: Swarm best objective change less than {:}'\
                        .format(minfunc))
                    if particle_output:
                        return p_min, fp[i_min], p, fp
                    else:
                        return p_min, fp[i_min]
                elif stepsize <= minstep:
                    print('Stopping search: Swarm best position change less than {:}'\
                        .format(minstep))
                    if particle_output:
                        return p_min, fp[i_min], p, fp
                    else:
                        return p_min, fp[i_min]
                else:
                    g = p_min.copy()
                    fg = fp[i_min]
    
            if debug:
                print('Best after iteration {:}: {:} {:}'.format(it, g, fg))
            it += 1
            Appenderint.append(it)
            Appenderg.append(g)
            Appenderfg.append(fg)
    
        print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))
        
        if not is_feasible(g):
            print("However, the optimization couldn't find a feasible design. Sorry")
        if particle_output:
            return g, fg, p, fp,Appenderint, Appenderg, Appenderfg
        else:
            return g, fg, Appenderint, Appenderg, Appenderfg
        
        
        
        
        
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    import seaborn as sns
    sns.set()

    from sklearn.model_selection import train_test_split
    
    
    
    
    
    
    
    
    
    
    #import pyswarms as pso
    #from pyswarms.utils.functions import single_obj as fx
    #from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
    
    #df_backtest_2 = df_backtest_h#vz.df_backtest_h2
    #df_backtest_2
    
    df_backtest_2 = Ausar
    
    param_A = df_backtest_2[df_backtest_2['Escenarios'] == 'B']#.iloc[5:-1,:]
    param_A = param_A.drop(columns = ['Fecha Inicial', 'Escenarios', 'Operacion','Resultado','Pips','Capital_Acm'])
    column_names = ['Volumen', 'SL aleatorio', 'TP Aleatorio', 'Capital']
    param_A = param_A.reindex(columns = column_names)
    
    targets = param_A['Capital']
    inputs = param_A.drop(['Capital'], axis = 1)
    
    
    x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.1, random_state=365)
    
    # Create a linear regression object
    reg = LinearRegression()
    # Fit the regression with the scaled TRAIN inputs and targets
    reg.fit(x_train,y_train)
    
    
    
    # Obtain the bias (intercept) of the regression
    reg.intercept_
    bint = reg.intercept_
    # Obtain the weights (coefficients) of the regression
    reg.coef_
    # Note that they are barely interpretable if at all
    
    # Create a regression summary where we can compare them with one-another
    reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
    reg_summary['Weights'] = reg.coef_
    reg_summary
        
    
    
    def maxiA(x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        return    reg_summary['Weights'][0]*x1 + reg_summary['Weights'][1]*x2 - reg_summary['Weights'][2]*x3 
        #return bint + -0.00798329*x1 + 41.7485*x2 + 8.16298*x3
    Alb = [1000, 3, 8]
    Aub = [3000, 5, 10]
    
    Bxopt, Bfopt, iters, xi, fmaxi = pso(maxiA, Alb, Aub) #Axopt : variables óptimas , Afopt: valor máximo, xi: variables ótimas a traves de la iteraciones
    
    Bfopt = Bfopt+bint
    fmaxi = fmaxi+(bint)
    
    plt.plot(np.arange(1,101,1), fmaxi)
    plt.ylabel('Capital')
    plt.xlabel('Numero de iteraciones')
    plt.title('Optimización contra iteraciones de capital')
    plt.show()
    
    return Bxopt, Bfopt, bint, reg_summary
#############################################################################
def Copt(Ausar):
    
    def _obj_wrapper(func, args, kwargs, x):
        return func(x, *args, **kwargs)
    
    def _is_feasible_wrapper(func, x):
        return np.all(func(x)>=0)
    
    def _cons_none_wrapper(x):
        return np.array([0])
    
    def _cons_ieqcons_wrapper(ieqcons, args, kwargs, x):
        return np.array([y(x, *args, **kwargs) for y in ieqcons])
    
    def _cons_f_ieqcons_wrapper(f_ieqcons, args, kwargs, x):
        return np.array(f_ieqcons(x, *args, **kwargs))
        
    def pso(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={}, 
            swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100, 
            minstep=1e-8, minfunc=1e-8, debug=False, processes=1,
            particle_output=False):
        """
        Perform a particle swarm optimization (PSO)
       
        Parameters
        ==========
        func : function
            The function to be minimized
        lb : array
            The lower bounds of the design variable(s)
        ub : array
            The upper bounds of the design variable(s)
       
        Optional
        ========
        ieqcons : list
            A list of functions of length n such that ieqcons[j](x,*args) >= 0.0 in 
            a successfully optimized problem (Default: [])
        f_ieqcons : function
            Returns a 1-D array in which each element must be greater or equal 
            to 0.0 in a successfully optimized problem. If f_ieqcons is specified, 
            ieqcons is ignored (Default: None)
        args : tuple
            Additional arguments passed to objective and constraint functions
            (Default: empty tuple)
        kwargs : dict
            Additional keyword arguments passed to objective and constraint 
            functions (Default: empty dict)
        swarmsize : int
            The number of particles in the swarm (Default: 100)
        omega : scalar
            Particle velocity scaling factor (Default: 0.5)
        phip : scalar
            Scaling factor to search away from the particle's best known position
            (Default: 0.5)
        phig : scalar
            Scaling factor to search away from the swarm's best known position
            (Default: 0.5)
        maxiter : int
            The maximum number of iterations for the swarm to search (Default: 100)
        minstep : scalar
            The minimum stepsize of swarm's best position before the search
            terminates (Default: 1e-8)
        minfunc : scalar
            The minimum change of swarm's best objective value before the search
            terminates (Default: 1e-8)
        debug : boolean
            If True, progress statements will be displayed every iteration
            (Default: False)
        processes : int
            The number of processes to use to evaluate objective function and 
            constraints (default: 1)
        particle_output : boolean
            Whether to include the best per-particle position and the objective
            values at those.
       
        Returns
        =======
        g : array
            The swarm's best known position (optimal design)
        f : scalar
            The objective value at ``g``
        p : array
            The best known position per particle
        pf: arrray
            The objective values at each position in p
       
        """
       
        assert len(lb)==len(ub), 'Lower- and upper-bounds must be the same length'
        assert hasattr(func, '__call__'), 'Invalid function handle'
        lb = np.array(lb)
        ub = np.array(ub)
        assert np.all(ub>lb), 'All upper-bound values must be greater than lower-bound values'
       
        vhigh = np.abs(ub - lb)
        vlow = -vhigh
    
        # Initialize objective function
        obj = partial(_obj_wrapper, func, args, kwargs)
        Appenderint = [] #appender
        Appenderg = []
        Appenderfg = []
        # Check for constraint function(s) #########################################
        if f_ieqcons is None:
            if not len(ieqcons):
                if debug:
                    print('No constraints given.')
                cons = _cons_none_wrapper
            else:
                if debug:
                    print('Converting ieqcons to a single constraint function')
                cons = partial(_cons_ieqcons_wrapper, ieqcons, args, kwargs)
        else:
            if debug:
                print('Single constraint function given in f_ieqcons')
            cons = partial(_cons_f_ieqcons_wrapper, f_ieqcons, args, kwargs)
        is_feasible = partial(_is_feasible_wrapper, cons)
    
        # Initialize the multiprocessing module if necessary
        if processes > 1:
            import multiprocessing
            mp_pool = multiprocessing.Pool(processes)
            
        # Initialize the particle swarm ############################################
        S = swarmsize
        D = len(lb)  # the number of dimensions each particle has
        x = np.random.rand(S, D)  # particle positions
        v = np.zeros_like(x)  # particle velocities
        p = np.zeros_like(x)  # best particle positions
        fx = np.zeros(S)  # current particle function values
        fs = np.zeros(S, dtype=bool)  # feasibility of each particle
        fp = np.ones(S)*np.inf  # best particle function values
        g = []  # best swarm position
        fg = np.inf  # best swarm position starting value
        
        # Initialize the particle's position
        x = lb + x*(ub - lb)
    
        # Calculate objective and constraints for each particle
        if processes > 1:
            fx = np.array(mp_pool.map(obj, x))
            fs = np.array(mp_pool.map(is_feasible, x))
        else:
            for i in range(S):
                fx[i] = obj(x[i, :])
                fs[i] = is_feasible(x[i, :])
           
        # Store particle's best position (if constraints are satisfied)
        i_update = np.logical_and((fx < fp), fs)
        p[i_update, :] = x[i_update, :].copy()
        fp[i_update] = fx[i_update]
    
        # Update swarm's best position
        i_min = np.argmin(fp)
        if fp[i_min] < fg:
            fg = fp[i_min]
            g = p[i_min, :].copy()
        else:
            # At the start, there may not be any feasible starting point, so just
            # give it a temporary "best" point since it's likely to change
            g = x[0, :].copy()
           
        # Initialize the particle's velocity
        v = vlow + np.random.rand(S, D)*(vhigh - vlow)
           
        # Iterate until termination criterion met ##################################
        it = 1
        while it <= maxiter:
            rp = np.random.uniform(size=(S, D))
            rg = np.random.uniform(size=(S, D))
    
            # Update the particles velocities
            v = omega*v + phip*rp*(p - x) + phig*rg*(g - x)
            # Update the particles' positions
            x = x + v
            # Correct for bound violations
            maskl = x < lb
            masku = x > ub
            x = x*(~np.logical_or(maskl, masku)) + lb*maskl + ub*masku
    
            # Update objectives and constraints
            if processes > 1:
                fx = np.array(mp_pool.map(obj, x))
                fs = np.array(mp_pool.map(is_feasible, x))
            else:
                for i in range(S):
                    fx[i] = obj(x[i, :])
                    fs[i] = is_feasible(x[i, :])
    
            # Store particle's best position (if constraints are satisfied)
            i_update = np.logical_and((fx < fp), fs)
            p[i_update, :] = x[i_update, :].copy()
            fp[i_update] = fx[i_update]
    
            # Compare swarm's best position with global best position
            i_min = np.argmin(fp)
            if fp[i_min] < fg:
                if debug:
                    print('New best for swarm at iteration {:}: {:} {:}'\
                        .format(it, p[i_min, :], fp[i_min]))
    
                p_min = p[i_min, :].copy()
                stepsize = np.sqrt(np.sum((g - p_min)**2))
    
                if np.abs(fg - fp[i_min]) <= minfunc:
                    print('Stopping search: Swarm best objective change less than {:}'\
                        .format(minfunc))
                    if particle_output:
                        return p_min, fp[i_min], p, fp
                    else:
                        return p_min, fp[i_min]
                elif stepsize <= minstep:
                    print('Stopping search: Swarm best position change less than {:}'\
                        .format(minstep))
                    if particle_output:
                        return p_min, fp[i_min], p, fp
                    else:
                        return p_min, fp[i_min]
                else:
                    g = p_min.copy()
                    fg = fp[i_min]
    
            if debug:
                print('Best after iteration {:}: {:} {:}'.format(it, g, fg))
            it += 1
            Appenderint.append(it)
            Appenderg.append(g)
            Appenderfg.append(fg)
    
        print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))
        
        if not is_feasible(g):
            print("However, the optimization couldn't find a feasible design. Sorry")
        if particle_output:
            return g, fg, p, fp,Appenderint, Appenderg, Appenderfg
        else:
            return g, fg, Appenderint, Appenderg, Appenderfg
        
        
        
        
        
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    import seaborn as sns
    sns.set()

    from sklearn.model_selection import train_test_split
    
    
    
    
    
    
    
    
    
    
    #import pyswarms as pso
    #from pyswarms.utils.functions import single_obj as fx
    #from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
    
    #df_backtest_2 = df_backtest_h#vz.df_backtest_h2
    #df_backtest_2
    
    df_backtest_2 = Ausar
    
    param_A = df_backtest_2[df_backtest_2['Escenarios'] == 'C']#.iloc[5:-1,:]
    param_A = param_A.drop(columns = ['Fecha Inicial', 'Escenarios', 'Operacion','Resultado','Pips','Capital_Acm'])
    column_names = ['Volumen', 'SL aleatorio', 'TP Aleatorio', 'Capital']
    param_A = param_A.reindex(columns = column_names)
    
    targets = param_A['Capital']
    inputs = param_A.drop(['Capital'], axis = 1)
    
    
    x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.1, random_state=365)
    
    # Create a linear regression object
    reg = LinearRegression()
    # Fit the regression with the scaled TRAIN inputs and targets
    reg.fit(x_train,y_train)
    
    
    
    # Obtain the bias (intercept) of the regression
    reg.intercept_
    bint = reg.intercept_
    # Obtain the weights (coefficients) of the regression
    reg.coef_
    # Note that they are barely interpretable if at all
    
    # Create a regression summary where we can compare them with one-another
    reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
    reg_summary['Weights'] = reg.coef_
    reg_summary
        
    
    
    def maxiA(x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        return    reg_summary['Weights'][0]*x1 + reg_summary['Weights'][1]*x2 + reg_summary['Weights'][2]*x3 
        #return bint + -0.00798329*x1 + 41.7485*x2 + 8.16298*x3
    Alb = [1000, 3, 7]
    Aub = [3000, 5, 9]
    
    Cxopt, Cfopt, iters, xi, fmaxi = pso(maxiA, Alb, Aub) #Axopt : variables óptimas , Afopt: valor máximo, xi: variables ótimas a traves de la iteraciones
    
    Cfopt = Cfopt+bint
    fmaxi = fmaxi+(bint)
    
    plt.plot(np.arange(1,101,1), fmaxi)
    plt.ylabel('Capital')
    plt.xlabel('Numero de iteraciones')
    plt.title('Optimización contra iteraciones de capital')
    plt.show()
    
    return Cxopt, Cfopt, bint, reg_summary



######################################3
    
def Dopt(Ausar):
    
    def _obj_wrapper(func, args, kwargs, x):
        return func(x, *args, **kwargs)
    
    def _is_feasible_wrapper(func, x):
        return np.all(func(x)>=0)
    
    def _cons_none_wrapper(x):
        return np.array([0])
    
    def _cons_ieqcons_wrapper(ieqcons, args, kwargs, x):
        return np.array([y(x, *args, **kwargs) for y in ieqcons])
    
    def _cons_f_ieqcons_wrapper(f_ieqcons, args, kwargs, x):
        return np.array(f_ieqcons(x, *args, **kwargs))
        
    def pso(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={}, 
            swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100, 
            minstep=1e-8, minfunc=1e-8, debug=False, processes=1,
            particle_output=False):
        """
        Perform a particle swarm optimization (PSO)
       
        Parameters
        ==========
        func : function
            The function to be minimized
        lb : array
            The lower bounds of the design variable(s)
        ub : array
            The upper bounds of the design variable(s)
       
        Optional
        ========
        ieqcons : list
            A list of functions of length n such that ieqcons[j](x,*args) >= 0.0 in 
            a successfully optimized problem (Default: [])
        f_ieqcons : function
            Returns a 1-D array in which each element must be greater or equal 
            to 0.0 in a successfully optimized problem. If f_ieqcons is specified, 
            ieqcons is ignored (Default: None)
        args : tuple
            Additional arguments passed to objective and constraint functions
            (Default: empty tuple)
        kwargs : dict
            Additional keyword arguments passed to objective and constraint 
            functions (Default: empty dict)
        swarmsize : int
            The number of particles in the swarm (Default: 100)
        omega : scalar
            Particle velocity scaling factor (Default: 0.5)
        phip : scalar
            Scaling factor to search away from the particle's best known position
            (Default: 0.5)
        phig : scalar
            Scaling factor to search away from the swarm's best known position
            (Default: 0.5)
        maxiter : int
            The maximum number of iterations for the swarm to search (Default: 100)
        minstep : scalar
            The minimum stepsize of swarm's best position before the search
            terminates (Default: 1e-8)
        minfunc : scalar
            The minimum change of swarm's best objective value before the search
            terminates (Default: 1e-8)
        debug : boolean
            If True, progress statements will be displayed every iteration
            (Default: False)
        processes : int
            The number of processes to use to evaluate objective function and 
            constraints (default: 1)
        particle_output : boolean
            Whether to include the best per-particle position and the objective
            values at those.
       
        Returns
        =======
        g : array
            The swarm's best known position (optimal design)
        f : scalar
            The objective value at ``g``
        p : array
            The best known position per particle
        pf: arrray
            The objective values at each position in p
       
        """
       
        assert len(lb)==len(ub), 'Lower- and upper-bounds must be the same length'
        assert hasattr(func, '__call__'), 'Invalid function handle'
        lb = np.array(lb)
        ub = np.array(ub)
        assert np.all(ub>lb), 'All upper-bound values must be greater than lower-bound values'
       
        vhigh = np.abs(ub - lb)
        vlow = -vhigh
    
        # Initialize objective function
        obj = partial(_obj_wrapper, func, args, kwargs)
        Appenderint = [] #appender
        Appenderg = []
        Appenderfg = []
        # Check for constraint function(s) #########################################
        if f_ieqcons is None:
            if not len(ieqcons):
                if debug:
                    print('No constraints given.')
                cons = _cons_none_wrapper
            else:
                if debug:
                    print('Converting ieqcons to a single constraint function')
                cons = partial(_cons_ieqcons_wrapper, ieqcons, args, kwargs)
        else:
            if debug:
                print('Single constraint function given in f_ieqcons')
            cons = partial(_cons_f_ieqcons_wrapper, f_ieqcons, args, kwargs)
        is_feasible = partial(_is_feasible_wrapper, cons)
    
        # Initialize the multiprocessing module if necessary
        if processes > 1:
            import multiprocessing
            mp_pool = multiprocessing.Pool(processes)
            
        # Initialize the particle swarm ############################################
        S = swarmsize
        D = len(lb)  # the number of dimensions each particle has
        x = np.random.rand(S, D)  # particle positions
        v = np.zeros_like(x)  # particle velocities
        p = np.zeros_like(x)  # best particle positions
        fx = np.zeros(S)  # current particle function values
        fs = np.zeros(S, dtype=bool)  # feasibility of each particle
        fp = np.ones(S)*np.inf  # best particle function values
        g = []  # best swarm position
        fg = np.inf  # best swarm position starting value
        
        # Initialize the particle's position
        x = lb + x*(ub - lb)
    
        # Calculate objective and constraints for each particle
        if processes > 1:
            fx = np.array(mp_pool.map(obj, x))
            fs = np.array(mp_pool.map(is_feasible, x))
        else:
            for i in range(S):
                fx[i] = obj(x[i, :])
                fs[i] = is_feasible(x[i, :])
           
        # Store particle's best position (if constraints are satisfied)
        i_update = np.logical_and((fx < fp), fs)
        p[i_update, :] = x[i_update, :].copy()
        fp[i_update] = fx[i_update]
    
        # Update swarm's best position
        i_min = np.argmin(fp)
        if fp[i_min] < fg:
            fg = fp[i_min]
            g = p[i_min, :].copy()
        else:
            # At the start, there may not be any feasible starting point, so just
            # give it a temporary "best" point since it's likely to change
            g = x[0, :].copy()
           
        # Initialize the particle's velocity
        v = vlow + np.random.rand(S, D)*(vhigh - vlow)
           
        # Iterate until termination criterion met ##################################
        it = 1
        while it <= maxiter:
            rp = np.random.uniform(size=(S, D))
            rg = np.random.uniform(size=(S, D))
    
            # Update the particles velocities
            v = omega*v + phip*rp*(p - x) + phig*rg*(g - x)
            # Update the particles' positions
            x = x + v
            # Correct for bound violations
            maskl = x < lb
            masku = x > ub
            x = x*(~np.logical_or(maskl, masku)) + lb*maskl + ub*masku
    
            # Update objectives and constraints
            if processes > 1:
                fx = np.array(mp_pool.map(obj, x))
                fs = np.array(mp_pool.map(is_feasible, x))
            else:
                for i in range(S):
                    fx[i] = obj(x[i, :])
                    fs[i] = is_feasible(x[i, :])
    
            # Store particle's best position (if constraints are satisfied)
            i_update = np.logical_and((fx < fp), fs)
            p[i_update, :] = x[i_update, :].copy()
            fp[i_update] = fx[i_update]
    
            # Compare swarm's best position with global best position
            i_min = np.argmin(fp)
            if fp[i_min] < fg:
                if debug:
                    print('New best for swarm at iteration {:}: {:} {:}'\
                        .format(it, p[i_min, :], fp[i_min]))
    
                p_min = p[i_min, :].copy()
                stepsize = np.sqrt(np.sum((g - p_min)**2))
    
                if np.abs(fg - fp[i_min]) <= minfunc:
                    print('Stopping search: Swarm best objective change less than {:}'\
                        .format(minfunc))
                    if particle_output:
                        return p_min, fp[i_min], p, fp
                    else:
                        return p_min, fp[i_min]
                elif stepsize <= minstep:
                    print('Stopping search: Swarm best position change less than {:}'\
                        .format(minstep))
                    if particle_output:
                        return p_min, fp[i_min], p, fp
                    else:
                        return p_min, fp[i_min]
                else:
                    g = p_min.copy()
                    fg = fp[i_min]
    
            if debug:
                print('Best after iteration {:}: {:} {:}'.format(it, g, fg))
            it += 1
            Appenderint.append(it)
            Appenderg.append(g)
            Appenderfg.append(fg)
    
        print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))
        
        if not is_feasible(g):
            print("However, the optimization couldn't find a feasible design. Sorry")
        if particle_output:
            return g, fg, p, fp,Appenderint, Appenderg, Appenderfg
        else:
            return g, fg, Appenderint, Appenderg, Appenderfg
        
        
        
        
        
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    import seaborn as sns
    sns.set()

    from sklearn.model_selection import train_test_split
    
    
    
    
    
    
    
    
    
    
    #import pyswarms as pso
    #from pyswarms.utils.functions import single_obj as fx
    #from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
    
    #df_backtest_2 = df_backtest_h#vz.df_backtest_h2
    #df_backtest_2
    
    df_backtest_2 = Ausar
    
    param_A = df_backtest_2[df_backtest_2['Escenarios'] == 'D']#.iloc[5:-1,:]
    param_A = param_A.drop(columns = ['Fecha Inicial', 'Escenarios', 'Operacion','Resultado','Pips','Capital_Acm'])
    column_names = ['Volumen', 'SL aleatorio', 'TP Aleatorio', 'Capital']
    param_A = param_A.reindex(columns = column_names)
    
    targets = param_A['Capital']
    inputs = param_A.drop(['Capital'], axis = 1)
    
    
    x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.1, random_state=365)
    
    # Create a linear regression object
    reg = LinearRegression()
    # Fit the regression with the scaled TRAIN inputs and targets
    reg.fit(x_train,y_train)
    
    
    
    # Obtain the bias (intercept) of the regression
    reg.intercept_
    bint = reg.intercept_
    # Obtain the weights (coefficients) of the regression
    reg.coef_
    # Note that they are barely interpretable if at all
    
    # Create a regression summary where we can compare them with one-another
    reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
    reg_summary['Weights'] = reg.coef_
    reg_summary
        
    
    
    def maxiA(x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        return    reg_summary['Weights'][0]*x1 + reg_summary['Weights'][1]*x2 + reg_summary['Weights'][2]*x3 
        #return bint + -0.00798329*x1 + 41.7485*x2 + 8.16298*x3
    Alb = [1000, 3, 7]
    Aub = [3000, 5, 9]
    
    Dxopt, Dfopt, iters, xi, fmaxi = pso(maxiA, Alb, Aub) #Axopt : variables óptimas , Afopt: valor máximo, xi: variables ótimas a traves de la iteraciones
    
    Dfopt = Dfopt+bint
    fmaxi = fmaxi+(bint)
    
    plt.plot(np.arange(1,101,1), fmaxi)
    plt.ylabel('Capital')
    plt.xlabel('Numero de iteraciones')
    plt.title('Optimización contra iteraciones de capital')
    plt.show()
    
    return Dxopt, Dfopt, bint, reg_summary


def captot():
   
    cap0 =  {'Escenario': ['A','B','C','D','Total', 'Capital Total Acumulado'], 'Ganacia' : [28.459, 811.506, 602.954,232.534,0,0]}
    df_captot = pd.DataFrame(cap0)
    df_captot.iloc[-1,-1] = 28.459 + 811.506 + 602.954 +232.534 + 100000
    df_captot.iloc[-2,-1] = 28.459 + 811.506 + 602.954 +232.534
    return df_captot
    
    