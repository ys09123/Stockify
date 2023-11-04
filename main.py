# Importing dashboard and finance libraries
import streamlit as st
import yfinance as yf
#importing scraping libraries
import requests
from bs4 import BeautifulSoup
# Importing datetime library
import datetime
from datetime import date, timedelta
# Importing pandas
import pandas as pd
# Importing libraries for visualization
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
# Importing modelling libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.title('Stock Price Prediction')
st.sidebar.info('Choose your options below:')

def main():
    option = st.sidebar.selectbox('Make a choice', ['Visualize', 'Recent Data', 'Predict'])
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Recent Data':
        dataframe()
    else:
        predict()

@st.cache_resource
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df

option = st.sidebar.text_input('Enter a Stock Symbol', value='AAPL')
option = option.upper()
# company_name = yf.Ticker(option).info['longName']
# st.subheader('Your selected company is:')
# st.subheader(company_name)
# stock_info = yf.Ticker(option)
# st.info(stock_info.info['longBusinessSummary'])
# st.write('***')
today = datetime.date.today()
duration = st.sidebar.number_input('Enter the duration (days)', value = 3650)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Enter Start Date', value = before)
end_date = st.sidebar.date_input('End Date', today)

if st.sidebar.button('Send'):
    if start_date < end_date:
        st.sidebar.success('Start Date: `%s`\n\nEnd Date: `%s`' %(start_date, end_date))
        download_data(option, start_date, end_date)
    
    else:
        st.sidebar.error('Error: End date must fall after start date')

data = download_data(option, start_date, end_date)
scaler = StandardScaler()

url = f"https://www.marketwatch.com/investing/stock/{option}/company-profile"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
name = soup.find('h1', class_='company__name')
if name is not None:
    name = name.get_text(strip=True)
    st.subheader(f"You selected:'{name}'")
else:
    print(" ")
    
element = soup.find('p', class_='description__text')
if element is not None:
    st.write(element.get_text(strip=True))
else:
    print(" ")
st.write("***")

def tech_indicators():
    st.header('Technical Indicators')
    option = st.radio('Choose a Technical Indicator to Visualize', ['Close', 'Bollinger Bands', 'Moving Average Convergence Divergence', 'Relative Strength Indicator', 'Simple Moving Average', 'Exponential Moving Average'])

    # Bollinger Bands
    bb_indicator = BollingerBands(data.Close)
    bb = data
    bb['bb_h'] = bb_indicator.bollinger_hband()
    bb['bb_l'] = bb_indicator.bollinger_lband()
    
    # Creating a new dataframe
    bb = bb[['Close', 'bb_h', 'bb_l']]

    # MACD
    macd = MACD(data.Close).macd()

    # RSI
    rsi = RSIIndicator(data.Close).rsi()

    # SMA
    sma = SMAIndicator(data.Close, window = 14).sma_indicator()

    # EMA
    ema = EMAIndicator(data.Close).ema_indicator()

    if option == 'Close':
        fig1 = px.line(data.Close, title='Close Price')
        st.plotly_chart(fig1, theme="streamlit", use_container_width=True)
        st.write('***')
        st.subheader("Indicator Information")
        st.info("The close price is the final price of a security at the end of a trading session. It is one of the most commonly used indicators in algorithmic trading.")

    elif option == 'Bollinger Bands':
        fig2 = px.line(bb, title='Bollinger Bands')
        st.plotly_chart(fig2, theme="streamlit", use_container_width=True)
        st.write('***')
        st.subheader("Indicator Information")
        st.info('Bollinger Bands is a tool used in stock trading to measure how much the price of a stock changes over time. It looks like three lines on a chart. The middle line is the average price of the stock, and the other two lines are above and below it. These two lines move closer together when the price of the stock is not changing much, and they move farther apart when the price of the stock is changing a lot.')

    elif option == 'Moving Average Convergence Divergence':
        fig3 = px.line(macd, title='MACD')
        st.plotly_chart(fig3, theme="streamlit", use_container_width=True)
        st.write('***')
        st.subheader("Indicator Information")
        st.info('Moving Average Convergence Divergence (MACD) is a tool used in finance to help people understand how the price of a stock or other investment might change over time. It is named after the person who created it, Gerald Appel. MACD is like a see-saw. The see-saw has two parts: a middle part and two outer parts. The middle part is the average price of the investment over a certain period of time, like 12 days. The outer parts are lines that are drawn above and below the middle part. These lines show how much the price of the investment might change over time. When the price of an investment is moving up, one side of the see-saw goes up and the other side goes down. When the price is moving down, the other side goes up and the first side goes down. This can help people understand how much risk there is in buying or selling an investment.')

    elif option == 'Relative Strength Indicator':
        fig4 = px.line(rsi, title='RSI')
        st.plotly_chart(fig4, theme="streamlit", use_container_width=True)
        st.write('***')
        st.subheader("Indicator Information")
        st.info('The Relative Strength Index (RSI) is a tool used in finance to help people understand how the price of a stock or other investment might change over time. It measures the speed and change of price movements and moves up and down between zero and 100. When the RSI is above 70, it generally indicates overbought conditions; when the RSI is below 30, it indicates oversold conditions. When the RSI is high, it means that people are buying a lot of shares of a company’s stock, and when it’s low, it means that people are selling more shares than they are buying.')


    elif option == 'Simple Moving Average':
        fig5 = px.line(sma, title='SMA')
        st.plotly_chart(fig5, theme="streamlit", use_container_width=True)
        st.write('***')
        st.subheader("Indicator Information")
        st.info('A simple moving average (SMA) is a tool used in finance to help people understand how the price of a stock or other investment might change over time. It is calculated by adding the most recent prices of an investment over a certain period of time, like 5 days, and then dividing the total by the number of days in that period.')

    else:
        fig6 = px.line(ema, title='EMA')
        st.plotly_chart(fig6, theme="streamlit", use_container_width=True)
        st.write('***')
        st.subheader("Indicator Information")
        st.info('An Exponential Moving Average (EMA) is a tool used in finance to help people understand how the price of a stock or other investment might change over time. It is a type of moving average (MA) that places a greater weight and significance on the most recent data points. The EMA is calculated by adding the most recent prices of an investment over a certain period of time, like 5 days, and then dividing the total by the number of days in that period. The EMA is different from a simple moving average (SMA) in that it places more weight on recent data points, which means it reacts more significantly to recent price changes than an SMA')

def dataframe():
    st.header('Recent Data')
    st.dataframe(data.tail(10))

def predict():
    model = st.radio('Choose a model:', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor'])
    num = st.number_input('How many days do you want the forecast for?', value = 5)
    num = int(num)
    if st.button('Predict'):
        if model == 'LinearRegression':
            engine = LinearRegression()
            model_engine(engine, num)

        elif model == 'RandomForestRegressor':
            engine = RandomForestRegressor()
            model_engine(engine, num)

        elif model == 'ExtraTreesRegressor':
            engine = ExtraTreesRegressor()
            model_engine(engine, num)

        elif model == 'KNeighborsRegressor':
            engine = KNeighborsRegressor()
            model_engine(engine, num)
        
        else:
            engine = XGBRegressor()
            model_engine(engine, num)
        
def model_engine(model, num):
    # getting only the closing price
    df = data[['Close']]

    # Shifting the closing price based on the number of days forecast
    df['preds'] = data.Close.shift(-num)

    # Scaling the data
    x = df.drop(['preds'], axis = 1).values
    x = scaler.fit_transform(x)

    # Storing the last num_days data
    x_forecast = x[-num:]

    # Selecting the required values for training
    x = x[:-num]

    # Getting the preds column
    y = df.preds.values

    # Selecting the required values for training
    y = y[:-num]

    # Splitting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

    # Training the model
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    # Predicting stock prices based on the number of days
    forecast_pred = model.predict(x_forecast)
    day = 1
    st.info('Prediction (including holidays and weekends):')
    for i in forecast_pred:
        # st.text(f'Day {day}: {i}')
        st.text(f'{date.today() + timedelta(days=day)} : {round(i, 2)}')
        day += 1

    st.caption(f'r2_score: {round(r2_score(y_test, preds), 2)}\n\nMAE: {round(mean_absolute_error(y_test, preds), 2)}')

if __name__ == '__main__':
    main()