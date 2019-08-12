from poloniex import Poloniex
import os
import time
from datetime import datetime, timedelta
import pandas as pd


def createColumns(ticker):
    return [ticker + '_w', ticker + '_o', ticker + '_c', ticker + '_l', ticker + '_h', ticker + '_v']

api_key = os.environ.get('V1E11WVF-0IVOC923-WN4Z7L1V-6GOM6QDS')
api_secret = os.environ.get('01d68b4b8a2e8f282c1c1e71824910a98f393c280c13fd2487dfe6045740d7e0d8343757b4a9a855ee813e0fafec10fbb7fa71c389865440c7bbfe8c718f6fef')
polo = Poloniex(api_key, api_secret)

hours = 2000
timestamp1 = time.mktime((datetime.now() - timedelta(days=0)).timetuple())
endTime = str(int(timestamp1))
startTime = time.mktime((datetime.now() - timedelta(hours=hours)).timetuple())

coin = ['USDT_BTC','USDT_ETH','USDT_QTUM','USDT_XRP','USDT_STR','USDT_LTC','USDT_BCHABC','USDT_ZEC','USDT_DASH','USDT_EOS','USDC_BTC','BTC_ETH','BTC_LTC','BTC_XRP','ETH_EOS','ETH_ZEC','ETH_ETC','XMR_LTC','XMR_ZEC','XMR_DASH']

data = pd.DataFrame()

data = data.append(pd.DataFrame(polo.returnChartData(coin[0], 300, startTime, endTime), columns=['close', 'volume']))
data = data.rename(columns={'close': coin[0] + '_c', 'volume': coin[0] + '_v'})
for i in range(1,len(coin)):
    data = data.join(pd.DataFrame(polo.returnChartData(coin[i], 300, startTime, endTime), columns=['close', 'volume']))
    data = data.rename(columns={'close': coin[i] + '_c', 'volume': coin[i] + '_v'})

#data2 = pd.DataFrame(data,columns=createColumns("USDT_BTC"))
#data = data.join(pd.DataFrame(polo.returnChartData(['USDT_ETH'],300,startTime,endTime),columns=createColumns("USDT_ETH")))
data.interpolate().to_csv('5minsData.csv')
