# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
#import plotly.graph_objects as go
#import plotly.express as px
import seaborn as sns
#import datetime
import streamlit as st
from PIL import Image


# import BPs
p_data = 'BP_Log.csv'
p_img = 'BP_ranges.png'

df = pd.read_csv(p_data)
df = df[~df['BP1'].isna()]


df['sys1'] = pd.DataFrame(df['BP1'].str.split(
    '/').to_list(), columns=['sys', 'delme'])['sys'].astype('float64')
df['sys2'] = pd.DataFrame(df['BP2'].str.split(
    '/').to_list(), columns=['sys', 'delme'])['sys'].astype('float64')
df['sys3'] = pd.DataFrame(df['BP3'].str.split(
    '/').to_list(), columns=['sys', 'delme'])['sys'].astype('float64')
df['mn_sys'] = (df['sys1'] + df['sys2'] + df['sys3']) / 3

dia1tmp = pd.DataFrame(df['BP1'].str.split(
    '/').to_list(), columns=['sys', 'delme'])['delme'].str.split('-').to_list()
df['di1'] = pd.DataFrame(dia1tmp)[0].astype('float64')
dia2tmp = pd.DataFrame(df['BP2'].str.split(
    '/').to_list(), columns=['sys', 'delme'])['delme'].str.split('-').to_list()
df['di2'] = pd.DataFrame(dia2tmp)[0].astype('float64')
dia3tmp = pd.DataFrame(df['BP3'].str.split(
    '/').to_list(), columns=['sys', 'delme'])['delme'].str.split('-').to_list()
df['di3'] = pd.DataFrame(dia3tmp)[0].astype('float64')

df['mn_di'] = (df['di1'] + df['di2'] + df['di3']) / 3


df['mn_cat'] = np.where((df['mn_sys'] > 180) | (df['mn_di'] > 120), 'Crisis',
                        np.where((df['mn_sys'] >= 140) | (df['mn_di'] >= 90), 'Stage 2',
                        np.where((df['mn_sys'] >= 130) | (df['mn_di'] >= 80), 'Stage 1',
                                 np.where((df['mn_sys'] >= 120) & (df['mn_di'] < 80), 'Elevated',
                                 np.where((df['mn_sys'] < 120) & (df['mn_di'] < 80), 'Normal', 'ERROR')))))

df['BP1_cat'] = np.where((df['sys1'] > 180) | (df['di1'] > 120), 'Crisis',
                         np.where((df['sys1'] >= 140) | (df['di1'] >= 90), 'Stage 2',
                                  np.where((df['sys1'] >= 130) | (df['di1'] >= 80), 'Stage 1',
                                           np.where((df['sys1'] >= 120) & (df['di1'] < 80), 'Elevated',
                                                    np.where((df['sys1'] < 120) & (df['di1'] < 80), 'Normal', 'ERROR')))))

df['BP2_cat'] = np.where((df['sys2'] > 180) | (df['di2'] > 120), 'Crisis',
                         np.where((df['sys2'] >= 140) | (df['di2'] >= 90), 'Stage 2',
                                  np.where((df['sys2'] >= 130) | (df['di2'] >= 80), 'Stage 1',
                                           np.where((df['sys2'] >= 120) & (df['di2'] < 80), 'Elevated',
                                                    np.where((df['sys2'] < 120) & (df['di2'] < 80), 'Normal', 'ERROR')))))

df['BP3_cat'] = np.where((df['sys3'] > 180) | (df['di3'] > 120), 'Crisis',
                         np.where((df['sys3'] >= 140) | (df['di3'] >= 90), 'Stage 2',
                                  np.where((df['sys3'] >= 130) | (df['di3'] >= 80), 'Stage 1',
                                           np.where((df['sys3'] >= 120) & (df['di3'] < 80), 'Elevated',
                                                    np.where((df['sys3'] < 120) & (df['di3'] < 80), 'Normal', 'ERROR')))))


df['mn_cdn'] = np.where((df['mn_sys'] >= 135) | (df['mn_di'] >= 85), 'High Risk',
                        np.where((df['mn_sys'] >= 121) | (df['mn_di'] > 80), 'Medium Risk', 'Normal'))

df['BP1_cdn'] = np.where((df['sys1'] >= 135) | (df['di1'] >= 85), 'High Risk',
                         np.where((df['sys1'] >= 121) | (df['di1'] > 80), 'Medium Risk', 'Normal'))

df['BP2_cdn'] = np.where((df['sys2'] >= 135) | (df['di2'] >= 85), 'High Risk',
                         np.where((df['sys2'] >= 121) | (df['di2'] > 80), 'Medium Risk', 'Normal'))

df['BP3_cdn'] = np.where((df['sys3'] >= 135) | (df['di3'] >= 85), 'High Risk',
                         np.where((df['sys3'] >= 121) | (df['di3'] > 80), 'Medium Risk', 'Normal'))


# df = df[['Date', 'DoW', 'sys1', 'di1', 'sys2', 'di2', 'sys3', 'di3', 'mn_sys', 'mn_di', 'mn_cat', 'BP1_cat', 'BP2_cat', 'BP3_cat','mn_cdn', 'BP1_cdn', 'BP2_cdn', 'BP3_cdn']]


s1 = df[['Date', 'sys1', 'di1', 'BP1_cat']]
s2 = df[['Date', 'sys2', 'di2', 'BP2_cat']]
s3 = df[['Date', 'sys3', 'di3', 'BP3_cat']]

s1.columns = ['Date', 'Systolic', 'Diastolic', 'Category']
s2.columns = ['Date', 'Systolic', 'Diastolic', 'Category']
s3.columns = ['Date', 'Systolic', 'Diastolic', 'Category']

bp_series = pd.concat([s1, s2, s3], axis=0)
# bp_series.columns = ['Date', 'Systolic', 'Diastolic', 'Category']


lst_date = pd.to_datetime(bp_series['Date'].max())
thirty_days_ago = lst_date - pd.to_timedelta(30, unit='d')

thirty_day = bp_series[pd.to_datetime(
    bp_series['Date']) > (pd.to_datetime(thirty_days_ago))]
thirty_med_sys = thirty_day['Systolic'].median()
thirty_med_di = thirty_day['Diastolic'].median()
thirty_avg_sys = thirty_day['Systolic'].mean()
thirty_avg_di = thirty_day['Diastolic'].mean()


prior_thirty = lst_date - pd.to_timedelta(30, unit='d')
prior_sixty = lst_date - pd.to_timedelta(60, unit='d')
prior = bp_series[(pd.to_datetime(bp_series['Date']) > prior_sixty) & (
    pd.to_datetime(bp_series['Date']) <= prior_thirty)]

prior_med_sys = prior['Systolic'].median()
prior_med_di = prior['Diastolic'].median()
prior_avg_sys = prior['Systolic'].mean()
prior_avg_di = prior['Diastolic'].mean()


#################################################################################################################
#################################################################################################################
#################################################################################################################
# Build Dashboard


add_sidebar = st.sidebar.selectbox(
    'Blood Pressure Results', ('Summary', 'Analysis', 'Raw Data'))

# summary
if add_sidebar == 'Summary':
    st.markdown('# Blood Pressure Summary')
    st.markdown(f"### Rolling 30-Day Averages current as of {lst_date:%Y-%m-%d}:")
    st.markdown('---')
#    st.markdown('### Enter small summmary test here ...')
    col1, col2, col3, col4, col5 = st.columns(5)  # [1.5,1.5,1,1.5,1.5]
    col1.markdown('#### Last 30 Median')
    col1.metric(label='Median Systolic:', value=int(round(thirty_med_sys, 0)), delta=(
        int(round(thirty_med_sys - 120))), delta_color='inverse')
    col1.metric(label='Median Diastolic:', value=int(round(thirty_med_di)), delta=(
        int(round(thirty_med_di - 80))), delta_color='inverse')
    col2.markdown('#### Last 30 Mean')
    col2.metric(label='Mean Systolic:', value=int(round(thirty_avg_sys, 0)), delta=(
        int(round(thirty_avg_sys - 120))), delta_color='inverse')
    col2.metric(label='Mean Diastolic:', value=int(round(thirty_avg_di)), delta=(
        int(round(thirty_avg_di - 80))), delta_color='inverse')
    col4.markdown('#### Prior 30 Median')
    col4.metric(label='Median Systolic:', value=int(round(prior_med_sys, 0)), delta=(
        int(round(prior_med_sys - 120))), delta_color='inverse')
    col4.metric(label='Median Diastolic:', value=int(round(prior_med_di)), delta=(
        int(round(prior_med_di - 80))), delta_color='inverse')
    col5.markdown('#### Prior 30 Mean')
    col5.metric(label='Mean Systolic:', value=int(round(prior_avg_sys, 0)), delta=(
        int(round(prior_avg_sys - 120))), delta_color='inverse')
    col5.metric(label='Mean Diastolic:', value=int(round(prior_avg_di)), delta=(
        int(round(prior_avg_di - 80))), delta_color='inverse')

# summary 30 days seaborn
    
    palette_dict = {'Systolic': '#332288',
                    'Diastolic': '#661100'}
    st.markdown('---')
    st.markdown('##### Last 30 Days')

    fig, ax = plt.subplots(figsize=(20, 10))

    thirty_day['Date'] = pd.to_datetime(thirty_day['Date'])
    thirty_day.set_index('Date', inplace=True)
    ax = sns.lineplot(data=thirty_day, palette=palette_dict, label = 'large', legend=False)
    ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))
    plt.axhline(y=thirty_day['Systolic'].median(), c='r', linestyle='--')
    plt.axhline(y=thirty_day['Diastolic'].median(), c='r', linestyle='--')
    plt.tick_params(axis='both', labelsize=18)
    plt.rcParams['font.size'] = 18
    ax.set_xlabel('')
    st.pyplot(fig)


# summary 60 days seaborn
    st.markdown('---')
    st.markdown('##### Prior 30 Days')

    fig, ax = plt.subplots(figsize=(20, 10))

    prior['Date'] = pd.to_datetime(prior['Date'])
    prior.set_index('Date', inplace=True)
    ax = sns.lineplot(data=prior, palette=palette_dict, legend=False)
    ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))
    plt.axhline(y=prior['Systolic'].median(), c='r', linestyle='--')
    plt.axhline(y=prior['Diastolic'].median(), c='r', linestyle='--')
    plt.tick_params(axis='both', labelsize=18)
    plt.rcParams['font.size'] = 18
    ax.set_xlabel('')
    st.pyplot(fig)
    
    st.markdown('---')  


if add_sidebar == 'Analysis':

    st.markdown('# Blood Pressure Analysis')
 #  st.markdown(f"### Data Current as of {lst_date:%Y-%m-%d}:")
    
    st.markdown(f'### The analysis contained uses this reference to categorize each blood pressure reading taken as of {lst_date:%Y-%m-%d}:')
    st.markdown('---')
    img = Image.open(p_img)
    st.image(img, caption='American Heart Association Blood Pressure Readings')
    
    st.markdown('---')
    
    palette_dict = {'Normal':'limegreen', 
                'Elevated':'goldenrod', 
                'Stage 1':'blue', 
                'Stage 2':'darkorange', 
                'Crisis': 'r'}

    a1, a2 = st.columns(2)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.scatterplot(data=bp_series, x='Diastolic', y='Systolic', hue='Category',  legend=False, palette=palette_dict, s=150)
    plt.title('Individual Blood Pressure Readings', fontsize = 28)
    plt.tick_params(axis='both', labelsize=20)
    ax.set(xlim=(70, 100))
    ax.set(ylim=(110, 170))
    ax.set(xlabel='')
    ax.set(ylabel='')
    plt.xlim(70)
    
    a1.pyplot(fig)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.scatterplot(data=df, x='mn_di', y='mn_sys', hue='mn_cat', legend=True, palette=palette_dict, s=150)
    plt.legend(title='BP Category')
    plt.tick_params(axis='both', labelsize=20)
    ax.set(xlabel='')
    ax.set(ylabel='')
    plt.legend(loc='upper left', fontsize= 20)
    plt.title('Average Daily Blood Pressure Readings', fontsize = 28)
    ax.set(xlim=(70, 100))
    ax.set(ylim=(110, 170))
    a2.pyplot(fig)

##################################################################################################################

    aa1, aa2 = st.columns(2)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.scatterplot(data=thirty_day, x='Diastolic', y='Systolic', hue='Category',  legend=False, palette=palette_dict, s=150)
    plt.title('Individual Blood Pressure, Last 30-Days', fontsize = 28)
    plt.tick_params(axis='both', labelsize=20)
    ax.set(xlim=(70, 100))
    ax.set(ylim=(110, 170))
    ax.set(xlabel='')
    ax.set(ylabel='')
    plt.xlim(70)
    
    aa1.pyplot(fig)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.scatterplot(data=prior, x='Diastolic', y='Systolic', hue='Category',  legend=True, palette=palette_dict, s=150)
    plt.legend(title='BP Category')
    plt.tick_params(axis='both', labelsize=20)
    ax.set(xlabel='')
    ax.set(ylabel='')
    plt.legend(loc='upper left', fontsize= 20)
    plt.title('Average Daily Blood Pressure, Prior 30-Days', fontsize = 28)
    ax.set(xlim=(70, 100))
    ax.set(ylim=(110, 170))
    aa2.pyplot(fig)
##################################################################################################################

    st.markdown('---')    
    a3,a4 = st.columns(2)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.catplot(data=bp_series, x='Category', kind='count', palette=palette_dict)
    plt.title('Count of Readings by Category', fontsize = 14)
    ax.set(xlabel='')
    ax.set(ylabel='')
    plt.tick_params(axis='both', labelsize=12)

    a3.pyplot(ax)

    st.markdown('---')   

    #pp_df = bp_series[['sys', 'di', 'cat']].copy()
    #pp_df.columns=['Systolic', 'Diastolic', 'Category']
    #a4.markdown('##### Count of Readings by Distribution')
    #fig, ax = plt.subplots(figsize=(20, 20))
    #ax = sns.pairplot(bp_series, hue='Category')
    #plt.tick_params(axis='both', labelsize=12)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.catplot(data=thirty_day, x='Category', kind='count', palette=palette_dict)
    plt.title('Last 30-Days', fontsize = 14)
    ax.set(xlabel='')
    ax.set(ylabel='')
    plt.tick_params(axis='both', labelsize=12)

    a4.pyplot(ax)
    

    prior = bp_series[(pd.to_datetime(bp_series['Date']) > prior_sixty) & (
    pd.to_datetime(bp_series['Date']) <= prior_thirty)]


if add_sidebar == 'Raw Data':
    st.markdown('# Blood Pressure Readings')
    st.markdown(f"### Raw Data current as of {lst_date:%Y-%m-%d}:")
    st.markdown('---')
    
    r1, r2 = st.columns(2)
    r1.markdown('##### Individual Readings')
    r1.dataframe(bp_series.sort_index(ascending=False))
    
    
    agg_30day = df[['Date', 'mn_sys', 'mn_di', 'mn_cat']]
    agg_30day.columns = ['Date', 'Systolic', 'Diastolic', 'Category']
    agg_30day['Date'] = pd.to_datetime(agg_30day['Date'])
    agg_30day['Date'] =  agg_30day['Date'].dt.strftime('%Y/%m/%d')
    agg_30day.set_index('Date')
    r2.markdown('##### Daily Average Readings')
    r2.dataframe(agg_30day.sort_index(ascending=False).round(0))
    

    
    st.markdown('---') 
   
    st.markdown('##### Tukey Five-Number - All Blood Pressure Readings')
    st.dataframe(bp_series.describe().T)
    
    st.markdown('---') 
    st.markdown('##### Tukey Five-Number - Last 30-Day Readings')
    st.dataframe(thirty_day.describe().T)   
    st.markdown('---')