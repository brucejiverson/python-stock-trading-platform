
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

def test_data_stationarity(df:pd.DataFrame):
    """This method performs an ADF (automated dicky fuller) test on the transformed df. Code is borrowed from
    https://www.analyticsvidhya.com/blog/2018/09/non-stationary-time-series-python/
    Quote: If the test statistic is less than the critical value, we can reject the null
    hypothesis (aka the series is stationary). When the test statistic is greater
    than the critical value, we fail to reject the null hypothesis (which means
    the series is not stationary)."""

    # # These lines make the data stationary for stationarity testing
    # # log(0) = -inf. Some indicators have 0 values which causes problems w/ log
    transformed_df = df.copy()

    n_asset = 1
    for i in range(2*n_asset): #loop through prices and volumnes
        col = transformed_df.columns[i]
        transformed_df[col] = transformed_df[col] - transformed_df[col].shift(1, fillna=0)

    # transformed_df.drop(transformed_df.index[0], inplace = True)

    #this assumes that the stationary method used on the df is the same used in _get_state()
    # print("TRANSFORMED DATA: ")
    # print(transformed_df.head())
    # print(transformed_df.tail())

    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    index=['Test Statistic','p-value','#Lags Used','Number of Observations Used', 'Success']
    for col in transformed_df.columns:
        print('Results for ' + col)
        dftest = adfuller(transformed_df[col], autolag='AIC')
        success = dftest[0] < dftest[4]['1%']
        dfoutput = pd.Series([*dftest[0:4], success], index = index)
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s) conf.'%key] = value
        print (dfoutput)
        print(' ')


    fig, ax = plt.subplots(1, 1)  # Create the figure
    fig.suptitle('Transformed data', fontsize=14, fontweight='bold')

    for col in transformed_df.columns:
            transformed_df.plot(y=col, ax=ax)

    fig.autofmt_xdate()
