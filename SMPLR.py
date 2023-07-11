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
        ("BTC-USD.NS","APOLLOHOSP.NS","TATACONSUM.NS","TATASTEEL.NS","RELIANCE.NS","LT.NS","BAJAJ-AUTO.NS","WIPRO.NS","BAJAJFINSV.NS","KOTAKBANK.NS",
        "ULTRACEMCO.NS","BRITANNIA.NS","TITAN.NS","INDUSINDBK.NS","ICICIBANK.NS","ONGC.NS","NTPC.NS","ITC.NS","BAJFINANCE.NS","NESTLEIND.NS",
        "TECHM.NS","HDFCLIFE.NS"))
    
    if st.button('Predict'):
        df = get_stock_data(ticker)
        ## ------------------------------- PREDICTION LOGIC -------------------------------

        df = df[['Open', 'High', 'Low', 'Adj Close', 'Volume']]
        df['HL_PCT'] = (df['High'] - df['Adj Close']) / df['Adj Close'] * 100
        df['PCT_change'] = (df['Adj Close'] - df['Open']) / df['Open'] * 100
        df = df[['Adj Close', 'HL_PCT', 'PCT_change', 'Volume']]
        forecast_col = 'Adj Close'
        df.fillna(-99999, inplace=True)
        forecast_out = int(math.ceil(0.0001*len(df)))
        df['label'] = df[forecast_col].shift(-forecast_out)
        df.dropna(inplace=True)
        x = np.array(df.drop(['label'],1))
        y = np.array(df['label'])
        x = preprocessing.scale(x)
        x_lately = x[-forecast_out:]
        y = np.array(df['label'])

        #Splitting our dataset to Training and Testing dataset
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        #Fitting Linear Regression to the training set
        clf = LinearRegression(n_jobs=-1)
        clf.fit(x_train, y_train)

        #predicting the Test set result
        acc = clf.score(x_test, y_test)
        forecast_set = clf.predict(x_lately)

        #showing predicted results
        st.subheader("Your latest predicted closing price is: ")
        st.title(forecast_set)

    st.write('You selected:', ticker)


##------------------------------- DATA VISUALIZATION -------------------------------
elif selected == "Visualization":
    ticker = st.selectbox("Pick any stock or index to predict:" ,
        ("BTC-USD.NS"))
    if st.button('Show Dataframe'):
        st.dataframe(get_stock_data(ticker))

    style.use('ggplot')
    df = pd.read_csv('BTC-USD.csv', index_col='Date', parse_dates=True)
    df['Forecast'] = np.nan
    last_date = df.iloc[-1].name
    last_unix = last_date.timestamp()
    one_day = 86400
    next_unix = last_unix + one_day    

    for i in forecast_set:
        next_date = datetime.datetime.fromtimestamp(next_unix)
        next_unix += one_day
        df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

    rcParams['figure.figsize']=15,5

    df['Adj Close'].plot()
    df['Forecast'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()


##------------------------------- ACCURACY -------------------------------
elif selected == "Accuracy":
    st.title("Accuracy Evaluation Metrics")
    ticker = st.selectbox("Pick any stock or index to predict:" ,
        ("BTC-USD.NS","APOLLOHOSP.NS","TATACONSUM.NS","TATASTEEL.NS","RELIANCE.NS","LT.NS","BAJAJ-AUTO.NS","WIPRO.NS","BAJAJFINSV.NS","KOTAKBANK.NS",
        "ULTRACEMCO.NS","BRITANNIA.NS","TITAN.NS","INDUSINDBK.NS","ICICIBANK.NS","ONGC.NS","NTPC.NS","ITC.NS","BAJFINANCE.NS","NESTLEIND.NS",
        "TECHM.NS","HDFCLIFE.NS"))
    df = get_stock_data(ticker)

    df = df[['Open', 'High', 'Low', 'Adj Close', 'Volume']]
    df['HL_PCT'] = (df['High'] - df['Adj Close']) / df['Adj Close'] * 100
    df['PCT_change'] = (df['Adj Close'] - df['Open']) / df['Open'] * 100
    df = df[['Adj Close', 'HL_PCT', 'PCT_change', 'Volume']]
    forecast_col = 'Adj Close'
    df.fillna(-99999, inplace=True)
    forecast_out = int(math.ceil(0.0001*len(df)))
    df['label'] = df[forecast_col].shift(-forecast_out)
    df.dropna(inplace=True)
    x = np.array(df.drop(['label'],1))
    y = np.array(df['label'])
    x = preprocessing.scale(x)
    x_lately = x[-forecast_out:]
    y = np.array(df['label'])

    #Splitting our dataset to Training and Testing dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    #Fitting Linear Regression to the training set
    clf = LinearRegression(n_jobs=-1)
    clf.fit(x_train, y_train)

    #Predicting the Test set result
    forecast_set = clf.predict(x_lately)

    #Evaluating the model
    import sklearn.metrics as metrics
    acc = clf.score(x_test, y_test)
    r2 = metrics.r2_score(y_test, forecast_set)
    mae = metrics.mean_absolute_error(y_test, forecast_set)
    mse = metrics.mean_squared_error(y_test, forecast_set)
    rmse = mse**0.5
    
    col1, col2 = st.columns(2)
    
    col1.metric("Accuracy", acc, "±0.5%")
    col1.metric("R2 Score", r2, "±5%")
    col2.metric("Mean Absolute Error ", mae, "± 5%")
    col1.metric("Mean Squared Error", mse, "± 5%")
    col2.metric("Root Mean Squared Error", rmse, "± 5%")

        