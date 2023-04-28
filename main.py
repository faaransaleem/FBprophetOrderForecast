import warnings
import itertools
import numpy as np
import random
import statsmodels.api as sm
import pandas as pd
import datetime as dt
# prophet by Facebook
from prophet import Prophet
# time series analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas_gbq

stores_list = list(set(All_Sales_data['user_id']))
df=All_Sales_data.copy()
algo_time=str(dt.datetime.today())


for stores in stores_list:
    df=All_Sales_data[All_Sales_data['user_id']==stores]
    df=df.drop('user_id',axis=1)
    df=df.rename(columns={'date':'ds','orders':'y'})
    df=df.set_index('ds')
    end_date = str(dt.datetime.today() - dt.timedelta(days=7))
    mask1 = df[df.index <= end_date]
    mask2 = df[df.index > end_date]
    X_tr=mask1.reset_index() 
    X_tst = mask2.reset_index()
    print('1st Step   '+str(stores))
    print(df)
    prior_scale=[0.01,0.1,0.5,0.7,0.9]
    fourier_order=[3,5,9,12,15,7]
    seasonality=['daily','weekly']
    seasonality_mode=['additive','multiplicative']

    final_forecast=pd.DataFrame(columns=['ds','trend','yhat_lower','yhat_upper','trend_lower','trend_upper','daily'
            ,'daily_lower','daily_upper','holidays','holidays_lower','holidays_upper'
            ,'multiplicative_terms','multiplicative_terms_lower','multiplicative_terms_upper'
            ,'national','national_lower','national_upper','weekly','weekly_lower','weekly_upper'
            ,'additive_terms','additive_terms_lower','additive_terms_upper','yhat'])
    model_number=0
    for season in seasonality:
        for fourier in fourier_order:
            for scale in prior_scale:
                for mode in seasonality_mode:
                    #df=df.reset_index()
                    model_params=mode+" " +season+" "+str(fourier)+" "+str(scale)
                    model =Prophet(
                    # growth='linear'
                    seasonality_mode=mode
                            #  ,holidays=holidays
                            #   ,holidays_prior_scale=75
                             #  ,changepoint_prior_scale = 25
                            #   ,changepoints=changepoints


                               )

                    if season=='daily':
                        select_period=1
                    else:
                        select_period=7
                    #model.add_seasonality(name='weekly', period=7, fourier_order=3,prior_scale=20)
                    #model.add_seasonality(name='monthly', period=30.5, fourier_order=3,prior_scale=15)
                    model.add_seasonality(name=season, period=select_period ,fourier_order=fourier,prior_scale=scale)
                    model.fit(X_tr)
                    future = model.make_future_dataframe(freq='H',periods=24*7)
                    forecast = model.predict(future)
                    forecast['model']=model_number
                    forecast['model_params']=model_params
                    final_forecast=final_forecast.append(forecast)
                    model_number=model_number+1
    final=df.merge(final_forecast,how='left',on='ds')
    write_df=final[['ds','y','yhat','model','model_params']]
    write_df['ds']=write_df['ds'].astype('string') 
    write_df['y']=write_df['y'].astype('string') 
    write_df['yhat']=write_df['yhat'].astype('string') 
    write_df['model']=write_df['model'].astype('string') 
    write_df['model_params']=write_df['model_params'].astype('string') 
    
    print('2nd Step   '+str(stores))

    write_df.to_gbq(destination_table=table,
    project_id= project_id,
    if_exists='replace',
    table_schema=table_schema,credentials=credentials)
    print(stores)
    print('Data Loaded into temp_stg.tmp_stg_prophet_hourly_newmodel')
    print("Fetching kravemart Data from BIGQUERY")

    from google.oauth2 import service_account


    All_Sales_data['date'] = pd.to_datetime(All_Sales_data['date'], format='%Y-%m-%d %H:%M:%S')
    All_Sales_data.set_index('date')  

    print("Fetching kravemart data Completed from BIGQUERY")
    params=model_final_select['model_params'][0].split()
    df=df.reset_index()
    final_forecast=pd.DataFrame(columns=['ds','trend','yhat_lower','yhat_upper','trend_lower','trend_upper','daily'
        ,'daily_lower','daily_upper','holidays','holidays_lower','holidays_upper'
        ,'multiplicative_terms','multiplicative_terms_lower','multiplicative_terms_upper'
        ,'national','national_lower','national_upper','weekly','weekly_lower','weekly_upper'
        ,'additive_terms','additive_terms_lower','additive_terms_upper','yhat'])

    mode=params[0]
    season=params[1]
    fourier=int(params[2])
    scale=float(params[3])
    model =Prophet(
    # growth='linear'
    seasonality_mode=mode
            #  ,holidays=holidays
            #   ,holidays_prior_scale=75
             #  ,changepoint_prior_scale = 25
            #   ,changepoints=changepoints


               )

    if season=='daily':
        select_period=1
    else:
        select_period=7


    #model.add_seasonality(name='weekly', period=7, fourier_order=3,prior_scale=20)
    #model.add_seasonality(name='monthly', period=30.5, fourier_order=3,prior_scale=15)
    model.add_seasonality(name=season, period=select_period ,fourier_order=fourier,prior_scale=scale)
    model.fit(df)
    future = model.make_future_dataframe(freq='H',periods=24*7)
    forecast = model.predict(future)
    forecast['model']=model_number
    forecast['model_params']=model_params
    final_forecast=final_forecast.append(forecast)
    model_number=model_number+1
    
    write_df=final_forecast[['ds','yhat']]
    write_df['ds']=write_df['ds'].astype('string')  
    write_df['yhat']=write_df['yhat'].astype('string') 
    write_df['store_id']=str(stores)
    write_df['algo_time']=algo_time
    write_df['store_id']=write_df['store_id'].astype('string')
    write_df['algo_time']=write_df['algo_time'].astype('string')

    write_df.to_gbq(destination_table=table,
    project_id= project_id,
    if_exists='append',
    #table_schema=table_schema
    credentials=credentials)
   
