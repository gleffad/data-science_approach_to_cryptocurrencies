import math
import pandas_datareader as web
import numpy as np
import pandas as pd
import matplotlib

'''
gold = web.DataReader('GC=F', data_source='yahoo',start='2016-01-01', end='2020-06-08')
silver = web.DataReader('SI=F', data_source='yahoo',start='2016-01-01', end='2020-06-08')
crude_oil = web.DataReader('CL=F', data_source='yahoo', start='2016-01-01', end='2020-06-08')
natural_gas = web.DataReader('NG=F', data_source='yahoo', start='2016-01-01', end='2020-06-08')
euros_usd = web.DataReader('EURUSD=X', data_source='yahoo', start='2016-01-01', end='2020-06-08')

gold.to_csv('/Users/x/_/git/github/data-science_approach_to_cryptocurrencies/builded_datasets/gold_dataset.csv', index=False, sep=',')
silver.to_csv('/Users/x/_/git/github/data-science_approach_to_cryptocurrencies/builded_datasets/silver_dataset.csv', index=False, sep=',')
crude_oil.to_csv('/Users/x/_/git/github/data-science_approach_to_cryptocurrencies/builded_datasets/crude_oil_dataset.csv', index=False, sep=',')
natural_gas.to_csv('/Users/x/_/git/github/data-science_approach_to_cryptocurrencies/builded_datasets/natural_gas_dataset.csv', index=False, sep=',')
euros_usd.to_csv('/Users/x/_/git/github/data-science_approach_to_cryptocurrencies/builded_datasets/euro_usd.csv', index=False, sep=',')

print(gold)
print(silver)
print(crude_oil)
print(natural_gas)
print(euros_usd)
'''

bitcoin = web.DataReader('BTC-USD', data_source='yahoo',start='2015-01-01', end='2020-06-08')
bitcoin.to_csv('../builded_datasets/bitcoin_dataset.csv', index=False, sep=',')
print(bitcoin)
bitcoin.info()
bitcoin['Close'].plot()
