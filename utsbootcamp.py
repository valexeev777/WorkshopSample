import numpy as np
import datetime as dt

import pandas as pd
from pandas_datareader import data

import matplotlib.pyplot as plt

import requests
from bs4 import BeautifulSoup
from IPython.display import display, Math, Latex

import statsmodels.formula.api as smf 
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices

# When editing a module, and not wanting to restatrt kernel every time use:
# import importlib
# importlib.reload(bc)
# import utsbootcamp as bc


def PageBreakPDF():
	# from IPython.display import display, Math, Latex
	# Usage: bc.PageBreakPDF()
	# Adds a page break in PDF output when saving Jupyter Notebook to PDF via LaTeX
	display(Latex(r"\newpage"))

def my_function():
    print('Hello you.')
	
def my_function2(name):
    print(f"Hello {name}, is it me you're looking for?")

def my_function3(name):
    print(f"Hello {name.capitalize()}, is it me you're looking for?")

def my_function4(name='alex'):
    if isinstance(name,str):
        print(f"Hello {name.capitalize()}, is it me you're looking for?")
    else:
        print('Inputs must be strings')	

def price2ret(x,keepfirstrow=False):
	ret = x.pct_change()
	if keepfirstrow:
		ret.fillna(0, inplace=True)
	else:
		ret.drop([ret.index[0]], inplace=True)
	return ret
	
def price2cret(x):
	ret = x.pct_change()
	ret.fillna(0, inplace=True)
	cret=((1 + ret).cumprod() - 1)
	return cret

def get_yahoo_data(ticker_list,
          start=dt.datetime(2020, 1, 2),
          end=dt.datetime(2020, 4, 30),
          column='Adj Close',plot=False):
    """
    This function reads in market data from Yahoo
    for each ticker in the ticker_list.
    
    Parameters
    ----------
    ticker_list : dict
        Example ticker_list = {'INTC': 'Intel',
               'MSFT': 'Microsoft',
               'IBM': 'IBM',
               'BHP': 'BHP',
               'TM': 'Toyota',
               'AAPL': 'Apple',
               'AMZN': 'Amazon',
               'BA': 'Boeing',
               'QCOM': 'Qualcomm',
               'KO': 'Coca-Cola',
               'GOOG': 'Google',
               'SNE': 'Sony',
               'PTR': 'PetroChina'}
    
    start : datetime, str ['yyyy-mm-dd'] 
    
    end : datetime, str ['yyyy-mm-dd']
    
    column : str ['Open', 'Low', 'High', 'Close',
        'Adj Close' (default), 'Volume']

    Returns
    -------
    pd.DataFrame
        Pandas dataframe containing specified columns for all tickers requested
    """
    
    ticker = pd.DataFrame()

    for tick in ticker_list:
        prices = data.DataReader(tick, 'yahoo', start, end)
        closing_prices = prices[column]
        ticker[tick] = closing_prices
    
    if plot:
        ret = ticker.pct_change()
        ((1 + ret).cumprod() - 1).plot(title='Cumulative Returns',figsize=(12,8))
    
    return ticker

#############################################
# Econometrics
#############################################

def regplot(df,formula,xName,yName):
	reg = smf.ols(formula, data=df).fit()
	print(reg.summary())
	x=df[xName]
	y=df[yName]
	yhat=reg.fittedvalues
	fig, ax = plt.subplots(figsize=(10,8))
	ax.plot(x, y, 'o', label="Raw data")
	ax.plot(x, yhat, 'r--.', label="OLS estimate")
	ax.legend(loc='best');
	return reg

def JBtest(resid,a=0.05):
    	# Residuals as input (reg.resid) and significance (dafault=0.05) 
    	test = sms.jarque_bera(reg.resid)
    	JBpvalue=test[1]
    	print(f'Jarque-Bera test:')
    	if JBpvalue<=a:
        	print(f'\tp-value is {JBpvalue:.03f}\n\tReject the null hypothesis that residuals are normally distributed. \n\tResiduals are NOT normally distributed.')
    	else:
        	print(f'\tp-value is {JBpvalue:.03f}\n\tFail to reject the null hypothesis that residuals are normally distributed. \n\tResiduals ARE normally distributed. ')
    	return JBpvalue

def BPtest(reg,a=0.05):
    	# Regression model as input (reg.resid) and significance (dafault=0.05) 
    	test = sms.het_breuschpagan(result.resid, result.model.exog)
    	BPpvalue=test[1]
    	print(f'Breusch-Pagan test:')
    	if BPpvalue<=a:
        	print(f'\tp-value is {BPpvalue:.03f}\n\tReject the null hypothesis of homoskedasticity. \n\tThe variance of the errors from a regression IS DEPENDENT on the values of the independent variables.')
    	else:
        	print(f'\tp-value is {BPpvalue:.03f}\n\tFail to reject the null hypothesis of homoskedasticity. \n\tThe variance of the errors from a regression does not depend on the values of the independent variables. ')
    	return BPpvalue


def VIF(df,formula):
	y, X = dmatrices(formula, data=df, return_type="dataframe")
	vif = pd.DataFrame()
	vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
	vif["Variable"] = X.columns
	return vif

def SimulateXY(b0=1,b1=2,n=1000,muX=0,sdX=1,errRatio=0.5):
	# Simulate data
	# n=1000           	# number of observations
	# muX=0 		# mean of X
	# sdX=1			# sd of X	
	# b0=1        		# define desired intercept for the line
	# b1=2        		# define desired slope of the line
	# errRatio=0.5 		# residual error relative to volatility of X variable

	# Simulate x data:
	x=np.random.normal(loc=muX,scale=sdX,size=(n,1))

	# Simulate errors. Errors must be with zero mean, but you can make standard deviation more or less than standard deviation of x (try!)
	err=np.random.normal(loc=0,scale=sdX*errRatio,size=(n,1))

	# Calculate y data:
	y = b0 + b1*x + err    # observed data (with error)
	y_true = b0 + b1*x     # true data

	df=pd.DataFrame(data=np.hstack((x,y)), columns=['x','y'])                   # Option 1
	# df = pd.DataFrame(np.concatenate([x,y], axis=1), columns= ['x','y'])      # Option 2
	
	return df, y_true

























class HTMLTableParser:
       
        def parse_url(self, url):
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'lxml')
            return [(table['id'],self.parse_html_table(table))\
                    for table in soup.find_all('table')]  
    
        def parse_html_table(self, table):
            n_columns = 0
            n_rows=0
            column_names = []
    
            # Find number of rows and columns
            # we also find the column titles if we can
            for row in table.find_all('tr'):
                
                # Determine the number of rows in the table
                td_tags = row.find_all('td')
                if len(td_tags) > 0:
                    n_rows+=1
                    if n_columns == 0:
                        # Set the number of columns for our table
                        n_columns = len(td_tags)
                        
                # Handle column names if we find them
                th_tags = row.find_all('th') 
                if len(th_tags) > 0 and len(column_names) == 0:
                    for th in th_tags:
                        column_names.append(th.get_text())
    
            # Safeguard on Column Titles
            if len(column_names) > 0 and len(column_names) != n_columns:
                raise Exception("Column titles do not match the number of columns")
    
            columns = column_names if len(column_names) > 0 else range(0,n_columns)
            df = pd.DataFrame(columns = columns,
                              index= range(0,n_rows))
            row_marker = 0
            for row in table.find_all('tr'):
                column_marker = 0
                columns = row.find_all('td')
                for column in columns:
                    df.iat[row_marker,column_marker] = column.get_text()
                    column_marker += 1
                if len(columns) > 0:
                    row_marker += 1
                    
            # Convert to float if possible
            for col in df:
                try:
                    df[col] = df[col].astype(float)
                except ValueError:
                    pass
            
            return df