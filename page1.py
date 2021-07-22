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
                                         children='Page1'),
             
                    ])

if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1')