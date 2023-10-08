import pandas as pd
#import quandl
import math, datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.pylab import rcParams
import streamlit as st
#import yfinance as yf
from yahoo_fin.stock_info import get_data
from streamlit_option_menu import option_menu
import warnings
warnings.filterwarnings('ignore')

selected = option_menu(None,
    options=["Prediction", "Visualization", "Accuracy"],
    icons=["book", "graph", "envelope"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

def get_stock_data(ticker):
    df = get_data(ticker, start_date = None, end_date = None, index_as_date = False, interval ='1d')
    return df

##------------------------------- PREDICTION PAGE -------------------------------
if selected == "Prediction":
    st.title("Predictor")
    ticker = st.selectbox("Pick any stock or index to predict:" ,
        ("APOLLOHOSP.NS","TATACONSUM.NS","TATASTEEL.NS","RELIANCE.NS","LT.NS","BAJAJ-AUTO.NS","WIPRO.NS","BAJAJFINSV.NS","KOTAKBANK.NS",
        "ULTRACEMCO.NS","BRITANNIA.NS","TITAN.NS","INDUSINDBK.NS","ICICIBANK.NS","ONGC.NS","NTPC.NS","ITC.NS","BAJFINANCE.NS","NESTLEIND.NS",
        "TECHM.NS","HDFCLIFE.NS","HINDALCO.NS","BHARTIARTL.NS","CIPLA.NS","TCS.NS","ADANIENT.NS","HEROMOTOCO.NS","MARUTI.NS","COALINDIA.NS",
        "BPCL.NS","HCLTECH.NS","ADANIPORTS.NS","DRREDDY.NS","EICHERMOT.NS","ASIANPAINT.NS","GRASIM.NS","JSWSTEEL.NS","DIVISLAB.NS","TATACONSUM.NS",
        "SBIN.NS","HDFCBANK.NS","HDFC.NS","WIPRO.NS","UPL.NS","POWERGRID.NS","TATAPOWER.NS","TATAMOTORS.NS","SUNPHARMA.NS","HINDUNILVR.NS",
        "SBILIFE.NS","INFY.NS","AXISBANK.NS"))
    
    
    if st.button('Predict'):
        df = get_stock_data(ticker)
        ## ------------------------------- PREDICTION LOGIC -------------------------------
        # Data Cleaning
        mean = df['open'].mean()
        df['open'] = df['open'].fillna(mean)

        mean = df['high'].mean()
        df['high'] = df['high'].fillna(mean)

        mean = df['low'].mean()
        df['low'] = df['low'].fillna(mean)

        mean = df['close'].mean()
        df['close'] = df['close'].fillna(mean)

        X = df[['open','high','low']]
        y = df['close'].values.reshape(-1,1)
        
        #Splitting our dataset to Training and Testing dataset
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        #Fitting Linear Regression to the training set
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        
        #predicting the Test set result
        y_pred = reg.predict(X_test)
        o = df['open'].values
        h = df['high'].values
        l = df['low'].values

        n = len(df)
        
        pred = []
        for i in range(0,n):
            open = o[i]
            high = h[i]
            low = l[i]
            output = reg.predict([[open,high,low]])
            pred.append(output)

        pred1 = np.concatenate(pred)
        predicted = pred1.flatten().tolist()

        t = predicted[-1]
        st.subheader("Your latest predicted closing price is: ")
        st.title(t)

    st.write('You selected:', ticker)


##------------------------------- DATA VISUALIZATION -------------------------------
elif selected == "Visualization":
    ticker = st.selectbox("Pick any stock or index to predict:" ,
        ("APOLLOHOSP.NS","TATACONSUM.NS","TATASTEEL.NS","RELIANCE.NS","LT.NS","BAJAJ-AUTO.NS","WIPRO.NS","BAJAJFINSV.NS","KOTAKBANK.NS",
        "ULTRACEMCO.NS","BRITANNIA.NS","TITAN.NS","INDUSINDBK.NS","ICICIBANK.NS","ONGC.NS","NTPC.NS","ITC.NS","BAJFINANCE.NS","NESTLEIND.NS",
        "TECHM.NS","HDFCLIFE.NS","HINDALCO.NS","BHARTIARTL.NS","CIPLA.NS","TCS.NS","ADANIENT.NS","HEROMOTOCO.NS","MARUTI.NS","COALINDIA.NS",
        "BPCL.NS","HCLTECH.NS","ADANIPORTS.NS","DRREDDY.NS","EICHERMOT.NS","ASIANPAINT.NS","GRASIM.NS","JSWSTEEL.NS","DIVISLAB.NS","TATACONSUM.NS",
        "SBIN.NS","HDFCBANK.NS","HDFC.NS","WIPRO.NS","UPL.NS","POWERGRID.NS","TATAPOWER.NS","TATAMOTORS.NS","SUNPHARMA.NS","HINDUNILVR.NS",
        "SBILIFE.NS","INFY.NS","AXISBANK.NS"))
    if st.button('Show Dataframe'):
        st.dataframe(get_stock_data(ticker))
        


##------------------------------- ACCURACY -------------------------------
elif selected == "Accuracy":
    st.title("Accuracy Evaluation Metrics")
    ticker = st.selectbox("Pick any stock or index to predict:" ,
        ("APOLLOHOSP.NS","TATACONSUM.NS","TATASTEEL.NS","RELIANCE.NS","LT.NS","BAJAJ-AUTO.NS","WIPRO.NS","BAJAJFINSV.NS","KOTAKBANK.NS",
        "ULTRACEMCO.NS","BRITANNIA.NS","TITAN.NS","INDUSINDBK.NS","ICICIBANK.NS","ONGC.NS","NTPC.NS","ITC.NS","BAJFINANCE.NS","NESTLEIND.NS",
        "TECHM.NS","HDFCLIFE.NS","HINDALCO.NS","BHARTIARTL.NS","CIPLA.NS","TCS.NS","ADANIENT.NS","HEROMOTOCO.NS","MARUTI.NS","COALINDIA.NS",
        "BPCL.NS","HCLTECH.NS","ADANIPORTS.NS","DRREDDY.NS","EICHERMOT.NS","ASIANPAINT.NS","GRASIM.NS","JSWSTEEL.NS","DIVISLAB.NS","TATACONSUM.NS",
        "SBIN.NS","HDFCBANK.NS","HDFC.NS","WIPRO.NS","UPL.NS","POWERGRID.NS","TATAPOWER.NS","TATAMOTORS.NS","SUNPHARMA.NS","HINDUNILVR.NS",
        "SBILIFE.NS","INFY.NS","AXISBANK.NS"))
    df = get_stock_data(ticker)

    mean = df['open'].mean()
    df['open'] = df['open'].fillna(mean)

    mean = df['high'].mean()
    df['high'] = df['high'].fillna(mean)

    mean = df['low'].mean()
    df['low'] = df['low'].fillna(mean)

    mean = df['close'].mean()
    df['close'] = df['close'].fillna(mean)

    X = df[['open','high','low']]
    y = df['close'].values.reshape(-1,1)
    
    #Splitting our dataset to Training and Testing dataset
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #Fitting Linear Regression to the training set
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    #Evaluating the model
    import sklearn.metrics as metrics
    r2 = metrics.r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    
    col1, col2 = st.columns(2)
    
    col1.metric("R2 Score", r2, "±5%")
    col2.metric("Mean Absolute Error ", mae, "± 5%")
    col1.metric("Mean Squared Error", mse, "± 5%")
    col2.metric("Root Mean Squared Error", rmse, "± 5%")

        