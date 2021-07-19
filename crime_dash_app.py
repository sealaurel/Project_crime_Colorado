# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import dash
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
import dash_html_components as html
#from functions_all import *
from dash_application_functions import *

import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

color_discrete_map_,df_to_use,df_grouped_crime_against,df_crime_categories,df_grouped_county,df_grouped_zip=create_dataframes()



app = dash.Dash(__name__)#, external_stylesheets=external_stylesheets)


app.layout = html.Div(
                    children=[
    
                                 html.H1(style={'color': 'red',
                                               'font-size': '100px',
                                               'font-family':'Ariel',
                                               'textAlign':'center',
                                               'background-color': 'black',
                                               'border':'none',
                                                'padding':'5px',
                                                'textAlign': 'center',
                                                'box-shadow': '0 8px 16px 0 rgba(0,0,0,0.5), 0 6px 20px 0 rgba(0,0,0,1)',
                                                'height':'200px'},
                                         children='Crime in Colorado'),
             
                                 html.Div(
                                          children=[
    
                                                    html.Div(style={'border':'none',
                                                                    'padding':'5px',
                                                                    'textAlign': 'center',
                                                                    'box-shadow': '0 8px 16px 0 rgba(0,0,0,0.2), 0 6px 20px 0 rgba(0,0,0,0.19)',
                                                                    'width':'47%'},
                                                            id='menu',
                                                            children=[
                                                                html.H3(style={'color': 'black',
                                                                                'font-size': '20px',
                                                                                'font':'Ariel',
                                                                                'textAlign':'center'},
                                                                         children='Filter out by crime category and date'),
                                                                
                                                                
                                                                dcc.Dropdown(
                                                                     style={'color': 'black',
                                                                        'font-size': '20px',
                                                                        'font-family':'Ariel',
                                                                        'textAlign':'center',
                                                                        'margin-left': '130px',
                                                                        'width':'70%'},
                                                                     id='crime_categories_menu', 
                                                                     options=make_menu_options(list(df_crime_categories.offense_category_name.unique())),
                                                                     multi=True),
                                                                
                                                                html.H3(style={
                                                                            'font-size': '10px',
                                                                            'textAlign':'center'}),

                                                                dcc.DatePickerRange(
                                                                        id='my-date-picker-range',
                                                                        start_date=dt.date(2009, 1, 1),
                                                                        end_date=dt.date(2020, 1, 1),
                                                                        min_date_allowed=dt.date(2009, 1, 1),
                                                                        max_date_allowed=dt.date(2020, 1, 1)),
                                                                
                                                                html.H3(style={'font-size':'10px',
                                                                              'textAlign':'center'}),
                                                                
                                                                html.Button('Submit',id='submit', n_clicks=0,
                                                                             style={'background-color': '#e7e7e7',
                                                                                    'border': 'black',
                                                                                    'color': 'black',
                                                                                    'padding': '20px 40px',
                                                                                    'text-align': 'center',
                                                                                    'button-align': 'center',
                                                                                    'text-decoration': 'none',
                                                                                    'display': 'inline-block',
                                                                                    'font-size': '20px',
                                                                                    'margin': '8px 6px',
                                                                                    'border-radius': '12px',
                                                                                    'box-shadow': '0 8px 16px 0 rgba(0,0,0,0.2), 0 6px 20px 0 rgba(0,0,0,0.19)'}
                                                                            )
                                                  ],
                                                            
                                                            
                                                    ),

                                            html.Div(style={'border':'none',
                                                          'padding':'3px',
                                                          'align': 'left'},
                                                          children=[                                

                                                            dcc.Graph(
                                                                          id='crime_categories',
                                                                          figure=plot_crime(df_crime_categories))]),

                                            html.Div(style={'border':'none',
                                                          'padding':'3px',
                                                          'align': 'right',
                                                          'width':'48%',
                                                          'display':'inline-block'},
                                                          children=[                                

                                                            dcc.Graph(
                                                                id='crime_against',
                                                                figure=plot_crime_against(df_grouped_crime_against))]),

                                            html.Div(style={'border':'none',
                                                        'padding':'3px',
                                                        'align': 'left',
                                                        'width':'48%',
                                                        'display':'inline-block'},
                                                        children=[                                

                                                            dcc.Graph(
                                                                id='county_map',
                                                                figure=plot_county_map(df_grouped_county))]
                                                    ),
                                                    ]
                                            ),
                            ]
                    )

@app.callback(
    Output(component_id='crime_categories', component_property='figure'),
    [Input(component_id='submit', component_property='n_clicks')],
    [State(component_id='crime_categories_menu', component_property='value'),
     State(component_id='my-date-picker-range', component_property='start_date'),
     State(component_id='my-date-picker-range', component_property='end_date')])
def update_output_div(n_clicks, input_value, start_date, end_date):
    df=df_crime_categories.loc[((df_crime_categories.timestamp>start_date)&(df_crime_categories.timestamp<end_date))]
    return plot_crime(df, options=input_value)


if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1')