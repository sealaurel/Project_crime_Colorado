import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import gzip
import shutil
import os
import sqlite3
from sqlite3 import Error
import csv
from pathlib import Path
import subprocess
import io
import pickle
import json

import statsmodels
import statsmodels.tsa.api as tsa
import plotly.express as px
import plotly.io as pio
import math
from math import sqrt
import holidays
import pmdarima as pm

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error


from pmdarima.arima.stationarity  import ADFTest
from pmdarima.arima.utils import ndiffs
from pmdarima.arima.utils import nsdiffs

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)



# This part is from the notebook that creates the database

def display_csvfileDF(file_name, folder, encoding='UTF-8'):
    """This function is needed to easily display the dataframe from a csv file with all the columns names
    Arguments:
    file name: file name as a string
    folder name: subfolder in data folder, str
    encoding: encoding of the file, can be adjusted, default is UTF-8'"""

    df = pd.read_csv('data/'+folder+file_name, header=0, encoding='UTF-8')
    return df

def table_query(q, cur):
    """ Function to display slelect query results (from sqlite db tables) as a DF
    Argument: 
    q: query, string"""
    result=cur.execute(q).fetchall()
    if result==[]:
        print('Nothing was found')
        df=None

    else:
        df = pd.DataFrame(result)
        df.columns = [x[0] for x in cur.description]
    return df

def import_data_to_tables(db_path_name, list_of_files, replace_dir_name):
    """Function to run an import of csv files from a list into an sqlite database
    Arguments:
    db_path_name: path to a database, str
    list_of_files: list of files to import, with a path to their location as strings. Example: 'data/Ref_tables/nibrs_activity_type.csv', 
    replace_dir_name: the name of the directory the source files are in. From example above 'data/Ref_tables/'"""

    db_name = Path(db_path_name).resolve()

    for entry in list_of_files:
        csv_file = Path(entry).resolve()
        result = subprocess.run(['sqlite3',
                                 str(db_name),
                                 '-cmd',
                                 '.mode csv',
                                 '.import --skip 1 ' + str(csv_file).replace('\\','\\\\')
                                 + ' ' + entry.replace('.csv','').replace(replace_dir_name,'')],
                                capture_output=True)
    return

def create_filelist(dir_name,n=2):
    """Function to create a file list in a directory to use in other functions
    Arguments:
    dir_name: the full path to files' location
    n: a number of files to skip before starting to build a list (sometimes there are invisible files like .gitignore or similar), default=2"""
    files=[]
    files = os.listdir(dir_name)
    files=files[n:]
    files=[dir_name + s for s in files]
    return files

def copy_files(file_list, dir_out, dir_in):
    """Function to easily copy files from a list from one directory to another.
    Arguments
    file_list: list of files to copy, strings, no path necessary
    dir_out: original location, str
    dir_in: destination location, str """
    for entry in file_list:
        shutil.copyfile(dir_out+entry,dir_in+entry)

#This part is from the first scrubbing notebook
        
def create_new_table(old_table, new_table, list_of_columns_to_leave, cur, drop_rename=False):
    """Function to create a new table with select columns only. The reason to make
    this function is that sqlite3 below v3.35 does not support DROP COLUMN operation
    Arguments:
    old_table: the source table, str
    new_table: the destination table, str
    list_of_columns_to_leave: a list of column names as strings
    drop_rename: rename flag, if True, the old table is dropped and the new one is renamed with the old one's name.
                default set to False"""
    
    statement='CREATE TABLE '+ new_table + ' AS SELECT ' + ', '.join(list_of_columns_to_leave) + ' FROM ' + old_table
    cur.execute(statement)
    
    if drop_rename==True:
        
        statement1='DROP TABLE '+old_table
        statement2='ALTER TABLE '+new_table+' RENAME TO '+old_table
        cur.execute(statement1)
        cur.execute(statement2)
        q='SELECT * FROM ' + old_table
        df_temp=table_query(q, cur)
        
    else:
        q='SELECT * FROM ' + new_table
        df_temp=table_query(q, cur)
        
    return df_temp

def add_update_clmn(tbl_to_updt, tbl_to_use, clmn_to_add, clmn_tbl1_to_use, clmn_to_join, cur):
    """Function adds a new column to a table and fills in the values based on the reference table values
    Arguments:
    tbl_to_updt: table to update, str
    tbl_to_use: reference table, str
    clmn_to_add: new column, str
    clmn_tbl1_to_use: columns in tbl_to_use to use to fill in the values in the new column, str
    clmn_to_join: column to use to join two tables, str

    """
    statement1='ALTER TABLE ' + tbl_to_updt + ' ADD COLUMN ' + clmn_to_add
    #print(statement1)
    cur.execute(statement1)
    
    statement2='UPDATE '+tbl_to_updt+' SET '+clmn_to_add+'=(SELECT '+clmn_tbl1_to_use+\
    ' FROM '+tbl_to_use+' WHERE '+tbl_to_use+'.'+clmn_to_join+'='+tbl_to_updt+'.'+clmn_to_join+')'
    #print(statement2)
    cur.execute(statement2)
    
    q='SELECT * FROM '+tbl_to_updt
    #print(q)
    df=table_query(q, cur)
    return df  

def update_value(table, column, old_value, new_value,  cur):
    """Updates values in the column based on the old values
    Arguments
    table: table to update, string
    column: column to update, string
    old_value: old value, do not forget to put double quotes around single quotes
    new_value: new value, do not forget to put double quotes around single quotes
    Example: update_value('victim_main_tmp', 'sex_code', "'F'", "'Female'")"""
    
    statement='UPDATE '+table+' SET '+column+'='+new_value+' WHERE '+column+'='+old_value
    #print(statement)
    cur.execute(statement)
    q='SELECT * FROM '+table
    df=table_query(q, cur)
    return df

def remove_dups(old_table, new_table, conn, cur, drop_rename=False):
    """Function to remove duplicates from a table
    Arguments:
    old_table: the source table, str
    new_table: the destination table, str
    drop_rename: rename flag, if True, the old table is dropped and the new one is renamed with the old one's name.
    default set to False"""
    
    q='SELECT * from '+old_table
    df=table_query(q, cur)
    df=df.drop_duplicates()
    df.to_sql(name=new_table, con=conn)
    
    if drop_rename==True:
        
        statement1='DROP TABLE '+old_table
        statement2='ALTER TABLE '+new_table+' RENAME TO '+old_table
        cur.execute(statement1)
        cur.execute(statement2)
        q='SELECT * FROM ' + old_table
        df_temp=table_query(q, cur)
        
    else:
        q='SELECT * FROM ' + new_table
        df_temp=table_query(q, cur)
        
    return df_temp

# These functions are from part2

def empty_string_count(df):
    """Function to count empty strings and null values in na dataframe columns
    Arguments:
    df: a dataframe"""
    clmns_list=df.columns
    for clmn in clmns_list:
        num_empty_strng=len(df[df[clmn]==''])
        num_nulls=len(df[df[clmn].isnull()])
        print('Column {} empty string count: {}'.format(clmn,num_empty_strng))
        print('Column {} null values count: {}'.format(clmn,num_nulls))
        print('******************************************************')
    print('Total number of records in the dataframe: {}'.format(len(df)))


def check_stationarity(ts, label, window=52, plot=True, index=['Dickey-Fuller test results'], min_=600, max_=1050):
    """This function plots the rolling mean and the rolling standard deviation os a timeseries and prints out
    the results of a Dickey-Fuller test
    Arguments:
    ts: time-series
    window: rolling window, int, 52 is a default
    label: a label of the timeseries to be displayed in the legend
    index: index is needed to create a dataframe with all scalar values
    plot: a flag to plot the series
    min_, max_ limits of values on y axis; default min_=600, max_=1050 """
    
    tsw_sma_mean=ts.rolling(window).mean()
    tsw_sma_std=ts.rolling(window).std()
    
    #Dickey-Fuller test results
    
    stnry_test=adfuller(ts, autolag='AIC')
    dict_results={}
    columns=['T_value','P_value','Lags','Observations',
             'Critical value, 1%','Critical value, 5%','Critical value, 10%','Stationary?']
    values=[stnry_test[0],stnry_test[1],stnry_test[2],stnry_test[3]]
    values_=[]
    for key, value in stnry_test[4].items():
        values_.append(value)
    values_.append(stnry_test[1]<0.05)

    values.extend(values_) 
    dict_results=dict(zip(columns,values))
    df=pd.DataFrame(dict_results, index=index)

    #Plotting
   
    if plot:
        with plt.style.context('ggplot'):
       
            fig, ax = plt.subplots(figsize=(18,6))

            ax.plot(ts.index, ts.values, label=label)
            ax.plot(tsw_sma_mean.index,tsw_sma_mean.values, label='Rolling Mean')
            ax.plot(tsw_sma_std.index,tsw_sma_std.values, label='Standard Deviation')
            title='Rolling Mean and Rolling Standard Deviation, '+str(window)+' week window'
            ax.set_title(title, fontsize=22);
            ax.set_ylabel('Offense Counts', fontsize=21);
            ax.set_xlabel('Year', fontsize=21);
            ax.tick_params(axis='y', labelsize=16)
            ax.tick_params(axis='x', labelsize=16)
            ax.set_ylim(min_, max_);
            
            plt.legend(loc='best',  fontsize=15);
            plt.show()
    return df


def decomposing(ts):
    """Function displays decomposition plots of a times-series in a pretty way
    Arguments:
    ts: time-series"""
    matplotlib.rc_file_defaults()

    decomposition=seasonal_decompose(ts)
    fig=plt.figure();
    fig=decomposition.plot();
    fig.set_size_inches(15,10);
    fig.set_facecolor('lightgrey');
    fig.suptitle('Decomposition Plots', fontsize=20, color='r');
    return fig

    
def create_ts_dict(column, df_, freq='W' ):
    """Function to create a dictionary with timeseries of a category in a dataframe:
    Arguments
    column: a column in the dataframe, str
    df_: a dataframe to consider, should be indexed by a timestamp"""
    
    list_categories=df_[column].unique()
    TS_dict_name={}

    for category in list_categories:
        df=df_.groupby(column).get_group(category)
        TS_dict_name[category]=df.resample(freq).count()['offense_id'].rename(category)
#        TS_dict_name[category].rename(category)
    return TS_dict_name


def display_figure_w_TSs(ts1, ts2, label1, label2, title, n=2, ts3=None, ts4=None, label3=None, label4=None, min_=600, max_=1050, limit_=True):
    """Function displays up to 4 time-series on one plot in a pretty format
    n: number of time-series fo the plot, default n=2
    ts1, t2, t3, t4: time-series to display; default for t3 and t4 is None
    label1, label2, label3, label4 : corresponding labels for the legend; default for label3 and label4 is None
    min_, max_ limits of values on y axis; default min_=600, max_=1050
    limit_: flag indicating if limiting y values needed
    """
#    matplotlib.rc_file_defaults()

    with plt.style.context('ggplot'):
        fig, ax = plt.subplots(figsize=(18,6))

        ax.plot(ts1.index, ts1.values, label=label1, lw=2)
        ax.plot(ts2.index, ts2.values, label=label2, lw=2)
        if n==3:
            ax.plot(ts3.index, ts3.values, label=label3, lw=2)
        elif n==4:
            ax.plot(ts3.index, ts3.values, label=label3, lw=2)
            ax.plot(ts4.index, ts4.values, label=label4, lw=2)

        ax.set_title(title, fontsize=22);
        ax.set_ylabel('Offense Counts', fontsize=20);
        ax.set_xlabel('Year', fontsize=20);
        if limit_:
            ax.set_ylim(min_, max_);
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        plt.legend(loc='upper left', fontsize=15);
        plt.show();
    return fig

def exog_reg_timeframe(start_date, end_date):
    """Creating a exogenous regressors TS (US holidays) for forecasting
    Arguments
    start date: start of a forecasted period
    end date: end of a forecasted period"""

    hlds_for= pd.date_range(start=start_date,  end=end_date, freq='D')

    us_holidays_for=[]
    years_=range(pd.to_datetime(start_date).year,pd.to_datetime(end_date).year)
    for year in years_:
        for holiday in holidays.UnitedStates(years=year).items():
            us_holidays_for.append(holiday[0])

    holiday_column_for=[]
    for date_ in hlds_for:
        if date_ in us_holidays_for:
            holiday_column_for.append(1)
        else:
            holiday_column_for.append(0)

    df_holidays_for=pd.DataFrame(hlds_for)
    df_holidays_for['Holiday']=holiday_column_for
    df_holidays_for=df_holidays_for.rename(columns={0:'Timestamp'})
    df_holidays_for=df_holidays_for.set_index('Timestamp')
    ts_holidays_for_weekly=df_holidays_for.resample('W').sum()
    return ts_holidays_for_weekly



def diagnostics(model, figsize=(15,7)):
    """ Just a simple function to display a model's summary and plot diagnostic plots
    Arguments:
    model: a model to diagnose"""
    display(model.summary())
    with plt.style.context('ggplot'):
        model.plot_diagnostics(figsize=figsize);
    return
    
def plot_predictions(ts, model, title, steps=104, xmin='2009', xmax='2022', figsize=(15,7), egog_flag=False, exog=None):
    """The function plots prediction for a model and a dataset in a pretty format.
    Arguments:
    ts: timeseries
    model: the model
    title: title of the plot, str
    steps: steps to make a forecast for, default is 52 (weeks in one year, when
                                                        using exogeneous predictors be aware of their weeks,
                                                        there might be more or less)
    xmin:  the start year to plot, default '2009'
    xmax:  the end year to plot, default '2022'
    figsize: figure size to use, default is (15,7)
    egog_flag: flag if  the model has exogeneous predictor, defalt is False
    exog: exogeneous predictors, array or a df, default is None"""
    
    
    if egog_flag:
        forecast = model.get_forecast(steps=steps, exog=exog)
    else:
        forecast = model.get_forecast(steps=steps)

    forecast_conf = forecast.conf_int()

    # Plotting
    with plt.style.context('ggplot'):
        fig, ax = plt.subplots(figsize=figsize)
        ax = ts.plot(label='Known Data', figsize=figsize)
        forecast.predicted_mean.plot(ax=ax, label='Forecast')

        ax.fill_between(forecast_conf.index,
                        forecast_conf.iloc[:, 0],
                        forecast_conf.iloc[:, 1], color='g', alpha=0.25)

        ax.set_title(title, fontsize=20);

        ax.set_ylabel('Offense Counts', fontsize=18);
        ax.set_xlabel('Year', fontsize=18);
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_xlim(xmin, xmax);

        plt.legend(loc='upper left', fontsize=15);
        plt.show()
    return fig
 
    
def us_holidays_predictors_TS(start='1/1/2015',  end='12/31/2019', years=range(2015, 2020), freq='W'):
    """Function created a time-series with US holidays for being used as exogeneous predictos 
    for a period of time. Returns a time-series with a given frequency.
    Arguments:
    start: start date of the period, default is 1/1/2009
    start: end date of the periodr, default is 12/31/2019
    year: a range of years, defaulty is range(2009,2020)
    freq: frequency of resampling, default is 'W'"""
    
    hlds= pd.date_range(start=start,  end=end, freq='D')

    us_holidays=[]
    years_= years
    for year in years_:
        for holiday in holidays.UnitedStates(years=year).items():
            us_holidays.append(holiday[0])

    holiday_column=[]
    for date_ in hlds:
        if date_ in us_holidays:
            holiday_column.append(1)
        else:
            holiday_column.append(0)

    df_holidays=pd.DataFrame(hlds)
    df_holidays['Holiday']=holiday_column
    df_holidays=df_holidays.rename(columns={0:'Timestamp'})
    df_holidays=df_holidays.set_index('Timestamp')
    ts_holidays_weekly=df_holidays.resample(freq).sum()
    return ts_holidays_weekly  

# These functions are from part 3

def ACF_PACF_multiple(ts_dict):
    """Function to display ACFs and PACFs for time-series in a dictionary"""
    for crime, ts in ts_dict.items():
        with plt.style.context('ggplot'):
            fig, axes = plt.subplots(figsize=(15,4), ncols=2)
            #plt.rc("figure", figsize=(15,5))
            plot_acf(ts,title='ACF, '+crime,ax=axes[0])
            plot_pacf(ts,title='PACF, '+crime,ax=axes[1])
            plt.show();
        
def check_stationarity_multiple(ts_dict, window=52, plot=True):
    """This function plots the rolling mean and the rolling standard deviation os a timeseries and prints out
    the results of a Dickey-Fuller test for all timeseries in the dictionary
    
    Arguments:
    dict_: dictionary with timeframes
    window: rolling window, int, 52 is a default
    plot: a flag to plot the series or just do the frint out
    
    Output is a df with all the results"""

    
    
    index_list=[]
    for key, value in ts_dict.items():
        index_list.append(key)
    
    dict_test={}
    columns=['Crime Category','Critical Value','P-value','Lags','Observations',
            'Critical value, 1%','Critical value, 5%',
            'Critical value, 10%','Stationary?']
    
    for column in columns:
        dict_test[column]=''
        
    df_results=pd.DataFrame(dict_test, index=['Number'])
        
    dict_stationary={}
    dict_non_stationary_diff={}
    
    for crime, ts in ts_dict.items():
        tsw_sma_mean=ts.rolling(window).mean()
        tsw_sma_std=ts.rolling(window).std()
    
        #Dickey-Fuller test results
        stnry_test=adfuller(ts, autolag='AIC')
        values=[crime, stnry_test[0],stnry_test[1],stnry_test[2], stnry_test[3]]
        values_=[]
        for key, value in stnry_test[4].items():
            values_.append(value)
        values_.append(stnry_test[1]<0.05)
        values.extend(values_)
        
        df_results.loc[len(df_results)] = values

        if values[-1]:
            stationarity='Stationary'
            dict_stationary[crime]=ts
        else:
            stationarity='Non-stationary'
            dict_non_stationary_diff[crime]=ts.diff().dropna()
            
        if plot:
            with plt.style.context('ggplot'):
                fig, ax = plt.subplots(figsize=(12,4))
                
                #matplotlib.rc_file_defaults()
                ax.plot(ts.index, ts.values, label=crime + ', '+stationarity )
                ax.plot(tsw_sma_mean.index,tsw_sma_mean.values, label='Rolling Mean')
                ax.plot(tsw_sma_std.index,tsw_sma_std.values, label='Standard Deviation')
                title= crime+', '+str(window)+' week window'
                ax.set_title(title, fontsize=16);
                ax.set_ylabel('Offense Counts', fontsize=15);
                ax.set_xlabel('Year', fontsize=15);
                plt.legend(loc='best');
                plt.show()
   
    return df_results, dict_stationary, dict_non_stationary_diff

def map_choropleth_location(df, locations_field, locations_field_display_name, value_column, value_column_display_name,
                            geojson_file, featureid_key, title):
    fig=px.choropleth_mapbox(data_frame=df, locations=locations_field, geojson=geojson_file, color=value_column, 
                        mapbox_style='stamen-terrain', zoom=6.0, height=900, featureidkey=featureid_key, 
                        center={'lat': 39.5501, 'lon': -105.7821}, opacity=0.7,
                        color_continuous_scale=px.colors.sequential.YlOrRd,
                        title=title,
                        labels={locations_field: locations_field_display_name,  value_column: value_column_display_name},
                        template = "plotly_dark")
    fig.update_layout(
    font_family="Arial",
    font_size=16,
    font_color="white",
    title_font_family="Arial",
    title_font_color="white",
    title_font_size=20)
    
    fig.update_layout(
    title={
        'y':0.98,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
    })
    
    fig.show()
    return fig

def print_out_models(dictionary):
    for crime, dict_ in dictionary.items():
        print('********************************************************************\nOFFENSE CATEGORY: '+ crime)
        for key, value in dict_.items():
            if key=='final_model':
                print('\nTHE FINAL MODEL SUMMARY: \n')
                display(value.summary());
                display(value.plot_diagnostics(figsize=(15,7)));
            elif key=='predict_fig':
                print('\nPREDICTION FOR TRAIN AND TEST sets: \n')
                display(value);
            else:
                print('\nFORECAST: \n')
                display(value);
            plt.close() 
            
            
def plot_predictions_px(ts, model, title, steps=104, xmin='2009', xmax='2022', figsize=(15,7), egog_flag=False, exog=None):
    """The function plots prediction for a model and a dataset in a pretty format.
    Arguments:
    ts: timeseries
    model: the model
    title: title of the plot, str
    steps: steps to make a forecast for, default is 52 (weeks in one year, when
                                                        using exogeneous predictors be aware of their weeks,
                                                        there might be more or less)
    xmin:  the start year to plot, default '2009'
    xmax:  the end year to plot, default '2022'
    figsize: figure size to use, default is (15,7)
    egog_flag: flag if  the model has exogeneous predictor, defalt is False
    exog: exogeneous predictors, array or a df, default is None"""
    
    
    if egog_flag:
        forecast = model.get_forecast(steps=steps, exog=exog)
    else:
        forecast = model.get_forecast(steps=steps)

    forecast_conf = forecast.conf_int()

    # Plotting

    fig = go.Figure([
    go.Scatter(
        name='Crime Data',
        x=ts.index, y=ts.values,
        mode='lines',
        line=dict(color='rgb(31, 119, 180)'),
    ),
    go.Scatter(
        name='Upper Bound',
        x=forecast_conf.index, y=forecast_conf.iloc[:, 0],
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False
    ),
         
    go.Scatter(
        name='Forecast with Confidence Intervals',
        x=forecast.predicted_mean.index,
        y=forecast.predicted_mean.values,
        marker=dict(color='black'),
        line=dict(width=2),
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        showlegend=True), 
        
    go.Scatter(
        name='Lower Bound',
        x=forecast_conf.index, y=forecast_conf.iloc[:, 1],
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        showlegend=False
    )
])
    fig.update_layout(
    template='ggplot2',
    title=title
    )

    fig.update_layout(
        font_family="Calibri",
        font_color="black",
        font_size=20,
        legend_font_size=18,
        title_font_family="Calibri",
        title_font_size=36)

    fig.update_layout(width=1200,
                      height=600)

    fig.update_layout(
        title={
            'y':0.91,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        },
        yaxis = dict(
                title = 'Number of Offenses',zeroline=True,
                showline = True),
        xaxis = dict(
                title = 'Year',zeroline=True,
                showline = True),
                )
    return fig

def predictions_testset(ts1, ts2, model, steps=26, figsize=(15,7), xmin='2018', xmax='2020', egog_flag=False, exog=None):
    """Functions to predict test set with confidence intervals
    Arguments:
    ts1: training set timeseries
    ts2: test set timeseries
    model: predictive model
    steps: number of weeks to make predictions
    figsize: figure size
    xmin: begining of prediction interval
    xmax: end of prediction interval"""
    
    
    if egog_flag:
        forecast = model.get_forecast(steps=steps, exog=exog)
    else:
        forecast = model.get_forecast(steps=steps)

    forecast_conf = forecast.conf_int()

# Plotting
    with plt.style.context('ggplot'):
        fig, ax = plt.subplots(figsize=figsize)
        ax = ts1.plot(label='Known Training Data')
        ax = ts2.plot(label='Known Test Data', figsize=figsize)
        forecast.predicted_mean.plot(ax=ax, label='Test Set Prediction')

        ax.fill_between(forecast_conf.index,
                            forecast_conf.iloc[:, 0],
                            forecast_conf.iloc[:, 1], color='g', alpha=0.25)

        ax.set_title('Prediction for the Test Set with Confidence Intervals', fontsize=20);

        ax.set_ylabel('Offense Counts', fontsize=18);
        ax.set_xlabel('Year', fontsize=18);
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_xlim(xmin, xmax);

        plt.legend(loc='upper left', fontsize=15);
        plt.show()
        return fig