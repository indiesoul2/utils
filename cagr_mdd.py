# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 23:12:31 2018

@author: ATIV 9 Lite
"""
def MDD(x):
    mdd_list=x.copy()
    for i in range(len(x)):
        mdd_list[i] = (x[i] - max(x[:i+1]))/x[i]
    return mdd_list

def result(profit): # profit은 누적 수익(배수)의 series
    # CAGR 구하기
    td = float((profit.index[-1]-profit.index[0]).days)
    CAGR = 100* (profit[-1]**(365.0/td) - 1)
    
    # MDD구하기
    mdd_list=profit.copy()
    for i in range(len(profit)):
        mdd_list[i] = (profit[i] - max(profit[:i+1]))/profit[i]
    
    mdd = max(abs(mdd_list)) * 100

    return CAGR, mdd
    