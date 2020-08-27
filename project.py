import plotly
from flask import Flask, render_template, request, redirect, Markup
from flask import Flask, render_template
import lxml, ssl
import pymysql
from io import BytesIO
import plotly.offline as po
import base64
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import numpy as np
import pandas as pd
from plotly.graph_objs import Layout
from sqlalchemy import create_engine

sqlEngine = create_engine('mysql+pymysql://root:@127.0.0.1', pool_recycle=3600)
sqlEngine.execute("create database if not exists stock_Info")
sqlEngine.execute("use stock_Info")


# Base data
start = '2019-04-1'
end = '2020-05-1'
company = ''

app = Flask(__name__)


@app.route("/", methods=['POST', 'GET'])
def home():
    global company
    if request.method == 'POST':
        company = request.form['searchbox']
        stock = yf.download(company, start, end)
        dbConnection = sqlEngine.connect()
        try:
            frame = stock.to_sql(company, dbConnection, if_exists='fail')
        except ValueError as vx:
            print(vx)
        except Exception as ex:
            print(ex)
        else:
            print("Table %s created successfully." % company)
        finally:
            dbConnection.close()
    return render_template("main.html")


# RSI
@app.route("/RSI_Analysis")
def RSI():
    if company == '':
        return redirect('/')
    dbConnection = sqlEngine.connect()
    data = pd.read_sql("select * from stock_Info." + company, dbConnection)
    # Window length for moving average
    window_length = 14

    data = yf.download('AAPL', start, end)
    close = data['Adj Close']
    delta = close.diff()
    delta = delta[1:]  # first element is nan
    up = delta.copy()
    down = delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # Calculate the EWMA
    roll_up1 = up.ewm(span=window_length).mean()
    roll_down1 = down.abs().ewm(span=window_length).mean()

    # Calculate the RSI based on EWMA
    RS1 = roll_up1 / roll_down1
    RSI1 = 100.0 - (100.0 / (1.0 + RS1))
    roll_up2 = up.rolling(window_length).mean()
    roll_down2 = down.abs().rolling(window_length).mean()

    # Calculate the RSI based on SMA
    RS2 = roll_up2 / roll_down2
    RSI2 = 100.0 - (100.0 / (1.0 + RS2))

    RSI1.plot()
    RSI2.plot()
    plt.legend(labels=['RSI via EWMA', 'RSI via SMA'], loc='best')

    sio = BytesIO()
    plt.savefig(sio, format='png')
    data = base64.encodebytes(sio.getvalue()).decode()

    plt.title('RSI Analysis')  # figure name

    # figure save to binary file
    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()

    # convert matplotlib figure to HTML
    imb = base64.b64encode(plot_data)  # 对plot_data进行编码
    ims = imb.decode()
    imd = "data:image/png;base64," + ims
    plt.close()
    return render_template("RSI.html", img=imd)


# k-curve
@app.route("/k_curve")
def kcurve():
    if company == '':
        return redirect('/')
    dbConnection = sqlEngine.connect()
    df = pd.read_sql("select * from stock_Info." + company, dbConnection)
    # df = yf.download('AAPL', start, end)   #read dataset

    dates = []
    for x in range(len(df)):
        newdate = str(df.index[x])
        newdate = newdate[0:10]
        dates.append(newdate)

    # df['Date'] = dates

    # call the Candlestick function
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                                         open=df['Open'],
                                         high=df['High'],
                                         low=df['Low'],
                                         close=df['Close'])])
    aPlot = plotly.offline.plot(fig,
                                config={"displayModeBar": True},
                                show_link=False,
                                include_plotlyjs=False,
                                output_type='div')

    return render_template("k_curve.html", img=Markup(aPlot))
    # with open('plotly_graph.html', 'w') as f:
    #     f.write(fig.to_html(include_plotlyjs='cdn'))


# Volumn
@app.route("/Volume_Analysis")
def Volumn():
    if company == '':
        return redirect('/')
    dbConnection = sqlEngine.connect()
    df = pd.read_sql("select * from stock_Info." + company, dbConnection)
    # df = yf.download('AAPL', start, end)   #read dataset 

    dates = []
    for x in range(len(df)):
        newdate = str(df.index[x])
        newdate = newdate[0:10]
        dates.append(newdate)

    fig = px.bar(df, x=df['Date'], y=df['Volume'], title='Volume analysis')

    aPlot = plotly.offline.plot(fig,
                                config={"displayModeBar": True},
                                show_link=False,
                                include_plotlyjs=False,
                                output_type='div')

    return render_template("volume.html", img=Markup(aPlot))


# Test
@app.errorhandler(404)
def not_found(e):
    # defining function
    return render_template("404.html")


if __name__ == "__main__":
    app.run(debug=True)

