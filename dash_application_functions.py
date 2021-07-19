import pickle
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from pandas import Grouper
import datetime as dt
import plotly.io as pio
import json
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from functions_all import *


CO_county_json=json.load(open('data/CO_counties_geo.json', 'r'))

def make_menu_options(list_for_menu):
    """Making a list of dictionaries for menu choices out of a list of strings """
    options=[]
    for option in list_for_menu:
        options.append({'label': option, 'value': option})
    return options


def plot_crime(df, options=None):
    """Plots/replots crime with certain categories"""
    if options==None:
        rslt_df=df
    else:
        rslt_df = df[df['offense_category_name'].isin(options)]
    
    fig = px.line(rslt_df, x='timestamp', y='count', color='offense_category_name', 
                    color_discrete_map=create_dataframes()[0],
                    labels={ 'timestamp': 'Date',
                           'count': 'Number of Offenses',
                           'offense_category_name': 'Offense Category'},
                    title='<b>Number of Offenses in Different Crime Categories</b>',
                    template='ggplot2'
                 )

    fig.update_layout(width=1000,
                      height=850)

    fig.update_layout(
        font_family="Serif",
        font_color="black",
        font_size=18,
        legend_font_size=13,
        title_font_family="Serif",
        title_font_size=22)

    fig.update_layout(
        title={
            'y':0.94,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        })

    fig.update_layout(
        xaxis=dict(
           rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         step='all',
                         stepmode='backward',
                         label='All'),
                    dict(count=5,
                         step='year',
                         stepmode= 'backward',
                         label='5Y'),
                    dict(count=1,
                         step='year',
                         stepmode= 'backward',
                         label='1Y')
                ])),
            rangeslider=dict(
                visible=True
            ),
        )
    )
    return fig

def plot_crime_against(df):
    """Function to plot crimes aginst bar chart"""
    
    
    fig = px.bar(df, x='crime_against', y='number_of_offenses', color='crime_against',
             animation_frame='year', animation_group='primary_county',
             hover_name="primary_county", hover_data=['number_of_offenses'], template='ggplot2',
             title='<b>Crime Against Categories per Year and County</b>',
             labels={
                     'crime_against': 'Crime Against',
                     'number_of_offenses': 'Number of Offenses',
                     'primary_county': 'County an Offense Reported To',
                     'year': 'Year'})

    fig.update_layout(width=1000,
                      height=850,
                      bargap=0.05)
    fig.update_layout(
        font_family="Serif",
        font_color="black",
        font_size=18,
        title_font_family="Serif",
        title_font_size=22)

    fig.update_layout(
        title={
            'y':0.94,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        })
    return fig

def plot_county_map(df):
    """Function to plot cholopleth of CO counties"""
    fig=px.choropleth_mapbox(data_frame=df, locations='primary_county', geojson=CO_county_json,
                                color='number_of_offenses', mapbox_style='carto-positron', zoom=6.13,
                                featureidkey='properties.name',
                                animation_frame='year', center={'lat': 39.0, 'lon': -105.5}, opacity=0.7,
                                color_continuous_scale=px.colors.sequential.Blues,
                                title='<b>Number of Offenses per County</b>',
                                labels={'primary_county': 'County',
                                        'number_of_offenses': 'Number of Offenses'},
                                template='ggplot2')
    fig.update_layout(
        font_family="Serif",
        font_color="black",
        font_size=18,
        legend_font_size=13,
        title_font_family="Serif",
        title_font_size=22)

    fig.update_layout(width=1000,
                      height=850)

    fig.update_layout(
        title={
            'y':0.94,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        })
    return fig

def create_dataframes(freq='M'):
    """Creating all dataframes used in the app and returning them, prodive frequency to group crime categories"""
    # Loading the main dataframe
    with open('data/pickled_dataframes/df_full_clean.pickle', 'rb') as f:
        df_full=pickle.load(f)

    # Creating a df to work with moving forward
    df_to_use=df_full.copy()
    df_to_use['year']=df_to_use['timestamp'].dt.year
    df_to_use['offense_id']=1
    df_to_use.drop(columns=['incident_id','agency_id'], inplace=True)

    #Creating a dataframe with 'crime agains' info to display
    df_grouped_crime_against = df_to_use.groupby(['primary_county', 'crime_against', 'year']).agg({'offense_id': ['count']})
    df_grouped_crime_against.columns = ['number_of_offenses']
    df_grouped_crime_against = df_grouped_crime_against.reset_index()
    df_grouped_crime_against=df_grouped_crime_against.sort_values(by=['year'])


    # Creating 'crime categories' dataframe
    colors_dark24=px.colors.qualitative.Dark24
    colors_dark24=colors_dark24[:-1]
    crime_categories=['Assault Offenses', 'Larceny/Theft Offenses', 
     'Drug/Narcotic Offenses', 'Fraud Offenses',
     'Destruction/Damage/Vandalism of Property', 
     'Burglary/Breaking & Entering', 'Sex Offenses', 
     'Arson', 'Motor Vehicle Theft', 'Kidnapping/Abduction',
     'Weapon Law Violations', 'Robbery',
     'Pornography/Obscene Material', 'Counterfeiting/Forgery', 
     'Bribery', 'Stolen Property Offenses', 'Prostitution Offenses',
     'Homicide Offenses', 'Extortion/Blackmail',
     'Embezzlement', 'Gambling Offenses',
     'Human Trafficking', 'Animal Cruelty']

    color_discrete_map_=dict(zip(crime_categories, colors_dark24))

    df_crime_categories = df_to_use.groupby(['offense_category_name', 
                              pd.Grouper(key='timestamp',
                                         freq=freq)])['offense_category_name'].agg(['count']).reset_index()
    df_crime_categories = df_crime_categories.sort_values(by=['timestamp', 'count'])


    # Creating a dataframe with crime per county info
    df_grouped_county = df_to_use.groupby(['primary_county', 'year']).agg({'offense_id': ['count']})
    df_grouped_county.columns = ['number_of_offenses']
    df_grouped_county = df_grouped_county.reset_index()
    df_grouped_county=df_grouped_county.sort_values(by=['year'])


    #Creating a dataframe  with crime per zip code indo
    df_grouped_zip = df_to_use.groupby(['primary_county', 'icpsr_zip','year']).agg({'offense_id': ['count']})
    df_grouped_zip.columns = ['number_of_offenses']
    df_grouped_zip = df_grouped_zip.reset_index()
    df_grouped_zip=df_grouped_zip.sort_values(by=['year'])
    
    return color_discrete_map_, df_to_use, df_grouped_crime_against, df_crime_categories, df_grouped_county, df_grouped_zip