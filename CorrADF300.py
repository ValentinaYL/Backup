import operator
import numpy as np
import statsmodels.tsa.stattools as sts
import matplotlib.pyplot as plt
import tushare as ts
import pandas as pd



#stock_pool = ['601818', '601998', '601169', '002142', '601398', '601328', '600000', '601288', '601939', '600036', '000001', '600016', '601988', '601166']
max_pair = 20
start = '2016-01-01'
end = '2016-12-01'

def get_code(line1,line2):
    line = line2 - line1
    stock_pool = []
    sector = pd.read_csv("c:/try1/sector300.csv", index_col=0)
    sector_code = sector['code'][line1:line2]
    resectorcode = sector_code.reset_index(drop=True) #重置index

    for i in range(line):
        stock_pool.append(str(resectorcode[i]).zfill(6))  #在股票池中添加元素
    return stock_pool


def rankcorrelation(stock_pool, max_pair, start, end):
    stockPool = stock_pool
    stock_num = len(stockPool)
    rank = {}
    price_of_j = ts.get_hist_data('hs300', start=start, end=end).iloc[::-1].dropna()
    for i in range(stock_num):
        # get the price of stock from TuShare
        price_of_i = ts.get_hist_data(stockPool[i], start=start, end=end).iloc[::-1].dropna()
        if len(price_of_i) != 0:
            close_price_of_ij = pd.concat([price_of_i['close'], price_of_j['close']], axis=1)
            close_price_of_ij = close_price_of_ij.dropna()#去掉缺失值
            # change the column name in the dataFrame
            close_price_of_ij.columns = ['close_i', 'close_j']
            # calculate the daily return and drop the return of first day cause it is NaN.计算每日回报
            ret_of_i = ((close_price_of_ij['close_i'] - close_price_of_ij['close_i'].shift())/close_price_of_ij['close_i'].shift()).dropna()
            ret_of_j = ((close_price_of_ij['close_j'] - close_price_of_ij['close_j'].shift())/close_price_of_ij['close_j'].shift()).dropna()
            # calculate the correlation and store them in rank1
            if len(ret_of_i) == len(ret_of_j):
                correlation = np.corrcoef(ret_of_i.tolist(), ret_of_j.tolist())#计算每日回报的相关系数,求出的是矩阵，用correlation[0, 1]取出相关系数
                m = stockPool[i]
                rank[m] = correlation[0, 1]
                rank1 = sorted(rank.items(), key=operator.itemgetter(1))#排序 sorted(列表名，key = operator.iteemgetter(1)即根据第二个域排序 )
                #print(rank1)
            #print(rank1)
            potentialPair = [item[0] for item in rank1]
            potentialPair = potentialPair[-max_pair:]#取相关系数最大的MaxPair对
    potentialPair.reverse()
    return potentialPair


def adftest(potential_pair, start, end):
    potentialPair = potential_pair
    Rank = {}
    price_of_1 = ts.get_hist_data('hs300', start=start, end=end).iloc[::-1]
    price_of_1 = price_of_1.dropna()
    closeprice_of_1 = pd.DataFrame(price_of_1['close']/price_of_1['close'][0])
    for i in range(len(potentialPair)):
        n = str(potentialPair[i])
        price_of_2 = ts.get_hist_data(n, start=start, end=end).iloc[::-1]
        price_of_2 = price_of_2.dropna()
        closeprice_of_2 = pd.DataFrame(price_of_2['close']/price_of_2['close'][0])
        close_price_final = pd.merge(closeprice_of_1, closeprice_of_2, left_index=True, right_index=True)
        if len(closeprice_of_1) != 0 and len(closeprice_of_2) != 0:
            model = pd.ols(y=close_price_final['close_y'], x=close_price_final['close_x'], intercept=True)# perform ols on these two stocks
            spread = close_price_final['close_y'] - close_price_final['close_x']*model.beta['x']#.beta-只取x变量前系数
            sta = sts.adfuller(spread, 1)
            pair = n
            #Augmented Dickey-Fuller test 拒绝域{u<u0},当p<0.05代表平稳，原假设H0为序列不平稳
            if sta[1] < 0.05:
                print(pair)

stock_pool = get_code(0, 100)
#print(stock_pool)
PotentialPair = rankcorrelation(stock_pool, max_pair, start, end)
print(PotentialPair)
adftest(PotentialPair, start, end)