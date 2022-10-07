from multiprocessing.sharedctypes import Value
import streamlit as st
import pandas as pd 
from matplotlib import pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np 
import webbrowser
from contextlib import contextmanager
from io import StringIO
from datetime import date
from dateutil.relativedelta import relativedelta
import datetime
import statsmodels.formula.api as sm



st.title("Auto ML Modelling")
#st.image("data//stolt.jpg",width = 800)
nav = st.sidebar.radio("Navigation",["Model Run"])

if 'dataframeTrain' not in st.session_state:
    st.session_state['dataframe'] = []
if 'dataframeTest' not in st.session_state:
    st.session_state['dataframeTest'] = []
if 'model_ids' not in st.session_state:
    st.session_state['model_ids'] = []
if 'model_paths' not in st.session_state:
    st.session_state['model_paths'] = []
if 'my_model' not in st.session_state:
    st.session_state['my_model'] = None
if 'dependent' not in st.session_state:
    st.session_state['dependent'] = []
if 'independent' not in st.session_state:
    st.session_state['independent'] = []


if nav == "Model Run":
    uploaded_file = st.file_uploader("Upload your CSV File")
    if uploaded_file is not None:
        st.session_state.dataframeTrain = pd.read_csv(uploaded_file) 
    if st.checkbox("Show Dataset"):
        st.table(st.session_state.dataframeTrain.head(8))
        #st.session_state.dependent= st.selectbox("Dependent variables are:" ,st.session_state.dataframeTrain.columns)
        #st.session_state.independent= st.multiselect("Independent variables are:" ,st.session_state.dataframeTrain.columns)
        df=st.session_state.dataframeTrain
        ShipName= st.multiselect("Ship Names To Filter:" ,df.Shipname.unique())
        df=df[df['Shipname'].isin(ShipName)]
        df['Legfromreporttime']=pd.to_datetime(df['Legfromreporttime']).dt.date
        TrainFromdate=st.date_input("Enter train from date")
        Traintodate=st.date_input("Enter train to date")
        TestFromdate=st.date_input("Enter test from date")
        Testtodate=st.date_input("Enter test to date")
        #TrainMonth= st.number_input("Enter Train Period in months")
        #TestMonth= st.number_input("Enter Test Period in months")
        #six_months = date + relativedelta(months=-TrainMonth)
        #one_month = date + relativedelta(months=+TestMonth)
        train=df[(df['Legfromreporttime'].between(TrainFromdate,Traintodate))]
        test=df[(df['Legfromreporttime'].between(TestFromdate,Testtodate))]
        st.subheader('TRAIN DATASET')
        st.download_button(label="Download TRAIN data",data=train.to_csv(), file_name='Train.csv' ,mime='text/csv')
        st.subheader('TEST DATASET')
        st.download_button(label="Download TEST data ",data=test.to_csv(), file_name='Test.csv' ,mime='text/csv')
        
        st.header("Linear Modelling")
        if st.button('Run Linear Model'):
            model = sm.ols('Propperday ~ STWmehrs_Imputed + Meandraft', train).fit()
            st.write(model.params)
            predicted=model.predict(test)
            result = pd.concat([test,predicted], axis=1)
            r2=model.rsquared
            r2 = "{:.2f}".format(r2)
            str(predicted.sum())
            a=predicted.sum()
            b=test.Propperday.sum()
            percError=(abs(a-b)/b)*100
            st.write("R Squared value is ",r2)
            a=st.write("Sum of Predicted value","{:.2f}".format(predicted.sum()))
            b=st.write("Sum of actual values",str(test.Propperday.sum()))
            st.write("Percent Error","{:.2f}".format(percError))
            st.download_button(label="Download data as CSV",data=result.to_csv(), file_name='PredictedValues.csv' ,mime='text/csv')





