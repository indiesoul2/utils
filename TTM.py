# -*- coding: utf-8 -*-
"""
Created on Thu May  2 13:54:18 2019

@author: user
"""

from pandas import Series, DataFrame
import pandas as pd
#from pandas_datareader import data
from pandas.tseries.offsets import Day, MonthEnd
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import FinanceDataReader as fdr
import tqdm as tn


#--- Download Raw Data ---#

#tickers = ['SPY', 'IEV', 'EWJ', 'EEM', 'TLO', 'IEF', 'IYR', 'RWX', 'GLD', 'DBC']
symbols = {'kodex200':'069500','미국Sp500선':'143850','미국나스닥100':'133690','차이나CSI300':'192090','미국달러선물':'138230',
          '골드선물':'132030','국고채10년레버':'167860','국채3년':'114820','헬스케어':'143860','모멘텀':'147970'}
tickers = symbols.values()
start = '2016'

all_data = {}
for ticker in tickers:
    #all_data[ticker] = data.DataReader(ticker, 'google', start)
    all_data[ticker] = fdr.DataReader(ticker,start)

prices = pd.DataFrame({tic: data['Close'] for tic, data in all_data.items()})
prices = prices.fillna(method = 'ffill')
rets = prices.pct_change(1)


#--- Basic Option ---#

fee = 0.0030
lookback = 12
num = 5


#--- Find Endpoints of Month ---#

s = pd.Series(np.arange(prices.shape[0]), index=prices.index)
ep = s.resample("M").max()


#--- Create Weight Matrix using 12M Momentum ---#

wts = list()

for i in range(lookback, len(ep)) :
    ## prices.index[ep[i]]       check the calendar
    cumret = prices.iloc[ep[i]] / prices.iloc[ep[i-12]] - 1
    K = rankdata(-cumret) <= num
    
    wt = np.repeat(0.00, prices.shape[1], axis = 0)
    wt[K] = 1 / num
    wt = pd.DataFrame(data = wt.reshape(1,prices.shape[1]),
                      index = [prices.index[ep[i]]],
                      columns = [prices.columns])
    wts.append(wt)
    
wts = pd.concat(wts)

#--- Portfolio Return Backtest Function ---#

def ReturnPortfolio(R, weights):
    if R.isnull().values.any() :
        print("NA's detected: filling NA's with zeros")
        R[np.isnan(R)] = 0

    if R.shape[1] != weights.shape[1] :
        print("Columns of Return and Weight is not same")        ## Check The Column Dimension
               
    if R.index[-1] < weights.index[0] + pd.DateOffset(days=1) :
        print("Last date in series occurs before beginning of first rebalancing period")
           
    if R.index[0] < weights.index[0] :
        R = R.loc[R.index > weights.index[0] + pd.DateOffset(days=1)]   ## Subset the Return object if the first rebalance date is after the first date 
     
    bop_value = pd.DataFrame(data = np.zeros(shape = (R.shape[0], R.shape[1])), index = R.index, columns = [R.columns])
    eop_value = pd.DataFrame(data = np.zeros(shape = (R.shape[0], R.shape[1])), index = R.index, columns = [R.columns])
    bop_weights = pd.DataFrame(data = np.zeros(shape = (R.shape[0], R.shape[1])), index = R.index, columns = [R.columns])
    eop_weights = pd.DataFrame(data = np.zeros(shape = (R.shape[0], R.shape[1])), index = R.index, columns = [R.columns])
    
    bop_value_total = pd.DataFrame(data = np.zeros(shape = R.shape[0]), index = R.index)
    eop_value_total = pd.DataFrame(data = np.zeros(shape = R.shape[0]), index = R.index)
    ret = pd.DataFrame(data = np.zeros(shape = R.shape[0]), index = R.index)
                       
    end_value = 1   # The end_value is the end of period total value from the prior period
    
    k = 0
    
    for i in range(0 , len(weights) -1 ) :
        fm = weights.index[i] + pd.DateOffset(days=1)
        to = weights.index[i + 1]            
        returns = R.loc[fm : to, ]

        jj = 0
        
        for j in range(0 , len(returns) ) :
            if jj == 0 :
                bop_value.iloc[k, :] = end_value * weights.iloc[i, :]
            else :
                bop_value.iloc[k, :] = eop_value.iloc[k-1, :]
            #print("============weights.iloc============")
            #print(weights.iloc[i, :])
            #print(end_value)
            
            bop_value_total.iloc[k] = bop_value.iloc[k, :].sum()
                        
            # Compute end of period values
            eop_value.iloc[k, :] = (1 + returns.iloc[j, :].values) * bop_value.iloc[k, :]..values
            eop_value_total.iloc[k] = eop_value.iloc[k, :].sum()
            
            # Compute portfolio returns
            ret.iloc[k] = eop_value_total.iloc[k] / end_value - 1
            end_value = float(eop_value_total.iloc[k])
            
            # Compute BOP and EOP weights
            bop_weights.iloc[k, :] = bop_value.iloc[k, :] / float(bop_value_total.iloc[k])
            eop_weights.iloc[k, :] = eop_value.iloc[k, :] / float(eop_value_total.iloc[k])
    
            jj += 1
            k += 1
    
    result = {'ret' : ret, 'bop_weights' : bop_weights, 'eop_weights' : eop_weights}
    return(result)


#--- Calculate Portfolio Return & Turnover ---#
    
result = ReturnPortfolio(rets, wts)

portfolio_ret = result['ret']
turnover = pd.DataFrame((result['eop_weights'].shift(1) - result['bop_weights']).abs().sum(axis = 1))
portfolio_ret_net = portfolio_ret - (turnover * fee)     


#--- Calculate Cumulative Return ---#

def ReturnCumulative(R) :
    R[np.isnan(R)] = 0
    
    temp = (1+R).cumprod()-1
    print("Total Return: ", round(temp.iloc[-1, :], 4)) 
    return(temp)

port_cumret = ReturnCumulative(portfolio_ret_net)


#--- Calculate Drawdown ---#

def drawdown(R) :
    dd = pd.DataFrame(data = np.zeros(shape = (R.shape[0], R.shape[1])), index = R.index, columns = [R.columns])
    R[np.isnan(R)] = 0
    
    for j in range(0, R.shape[1]):
        
        if (R.iloc[0, j] > 0) :
            dd.iloc[0, j] = 0
        else :
            dd.iloc[0, j] = R.iloc[0, j]
            
        for i in range(1 , len(R)):
            temp_dd = (1+dd.iloc[i-1, j]) * (1+R.iloc[i, j]) - 1
            if (temp_dd > 0) :
                dd.iloc[i, j] = 0
            else:
                dd.iloc[i, j] = temp_dd
    
    return(dd)
    
port_dd = drawdown(portfolio_ret_net)


#--- Graph: Portfolio Return and Drawdown ---#

#fig, axes = plt.subplots(2, 1)
#port_cumret.plot(ax = axes[0], legend = None)
#port_dd.plot(ax = axes[1], legend = None)


#--- Daily Return Frequency To Yearly Return Frequency ---#

def apply_yearly(R) :
    
    s = pd.Series(np.arange(R.shape[0]), index=R.index)
    ep = s.resample("A").max()
    temp = pd.DataFrame(data = np.zeros(shape = (ep.shape[0], R.shape[1])), index = ep.index.year, columns = [R.columns])

    for i in range(0 , len(ep)) :
        if (i == 0) :
            sub_ret = R.iloc[ 0 : ep[i] + 1, :]
        else :
            sub_ret = R.iloc[ ep[i-1]+1 : ep[i] + 1, :]
        temp.iloc[i, ] = (1 + sub_ret).prod() - 1
    
    return(temp)

#yr_ret = apply_yearly(portfolio_ret_net)
#yr_ret.plot(kind = 'bar')
