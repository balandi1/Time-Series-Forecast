import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from matplotlib import pyplot
rcParams['figure.figsize'] = 20, 10
import pandas as pd
import numpy as np
import itertools
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from math import sqrt
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore") # ignore warning messages

# Define p, d, q parameters
p = d = q = range(0, 2)

# Get minimum p,d,q of all different combinations
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 7) for x in list(itertools.product(p, d, q))]

# Function for evaluating model for first prediction
def trainModel_1(productData):
    totalsum = np.sum(productData,axis=1).astype(float)
    sumarray = np.array(totalsum)
    ts = pd.Series(sumarray)
    ts.index = pd.to_datetime(ts.index,unit = 'D')
    train_data=productData[0:100]
    test_data=productData[100:118]
    sum_train = np.sum(train_data,axis=1).astype(float)
    sum_test = np.sum(test_data,axis=1).astype(float)
    sumarray_train = np.array(sum_train)
    sumarray_test = np.array(sum_test)
    ts_train = pd.Series(sumarray_train)
    ts_test = pd.Series(sumarray_test)
    ts_train.index = pd.to_datetime(ts_train.index,unit = 'D')
    ts_test.index = pd.to_datetime(ts_test.index,unit = 'D')
    eval_model = sm.tsa.statespace.SARIMAX(ts_train, order=(1,1,1),seasonal_order=(0,1,1,7),enforce_stationarity=False,enforce_invertibility=False)
    eval_result = eval_model.fit(disp=0)
    eval_foreCast = eval_result.forecast(steps = 18)
    #plt.plot(ts,label='Original Values')
    #plt.plot(eval_foreCast, color='red',label='Predicted Values' )
    #plt.legend()
    #plt.show()
    rms=sqrt(mean_squared_error(ts_test,eval_foreCast))
    #print(rms)
    return;

# Function for evaluating model for second prediction
def trainModel_2(productData):
    count=0
    train_data=productData[0:100]
    test_data=productData[100:118]
    for column in train_data:       
        productarray=np.array(train_data[column]).astype(float)
        seriesproduct=pd.Series(productarray)
        seriesproduct.index=pd.to_datetime(seriesproduct.index,unit='D')
        minval=0
        paramval=''
        seasonalparam=''
        for product_param in pdq:
            for product_param_seasonal in seasonal_pdq:
                mod = sm.tsa.statespace.SARIMAX(seriesproduct,
                                                order=product_param,
                                                seasonal_order=product_param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                product_results = mod.fit()
                if minval==0:
                    minval=product_results.aic
                    paramval=product_param
                    seasonalparam=product_param_seasonal
                if product_results.aic<minval:
                    minval=product_results.aic
                    paramval=product_param
                    seasonalparam=product_param_seasonal
                #print(product_minval,product_paramval,product_seasonalparam)
   
                # Apply SARIMAX for training
                testRow = sm.tsa.statespace.SARIMAX(seriesproduct,
                                         order=product_paramval,
                                         seasonal_order=product_seasonalparam,
                                         enforce_stationarity=False,
                                         enforce_invertibility=False)
                testArima = testRow.fit(disp=0)
    
                # Predict values for test data
                arimaForecast=testArima.forecast(steps=18)
                final=np.array(arimaForecast)
                
                testingproductarray1=np.array(testing[count]).astype(float)
                testingproductarray=testingproductarray1
                testingseriesproduct1=pd.Series(testingproductarray)
                testingseriesproduct=testingseriesproduct1
                testingseriesproduct.index=pd.to_datetime(testingseriesproduct.index,unit='D')
    

                # Find error between predicted and actual values
                rms1 = sqrt(mean_squared_error(testingseriesproduct, arimaForecast))
                #print(rms1)
                count=count+1
    return;


# Read the product distribution file
path = 'product_distribution_training_set.txt'
productData = pd.read_csv(path,sep='\t',header=None)

# Transpose the dataframe
productData = productData.T

# Remove the first row from data (key products row)
productHeader = productData.iloc[0]
productData = productData.drop(productData.index[[0]])

# Test the Model before prediction
#trainModel_1(productData)

print("Predicting overall sale for next 29 days...")

# Sum of all products sold at each day
sum = np.sum(productData,axis=1).astype(float)
sumarray = np.array(sum)

# Create a time series object with the sumarray(total products sold for each day)
ts = pd.Series(sumarray)

# Create the datetime stamp for the time series object
ts.index = pd.to_datetime(ts.index,unit = 'D')

# Pass the time series object to ARIMA model to train the test data
model = sm.tsa.statespace.SARIMAX(ts, order=(1,1,1),seasonal_order=(0,1,1,7),enforce_stationarity=False,enforce_invertibility=False)
results_ARIMA = model.fit(disp=0)

# Open a file called output.txt to write the predictions
file = open('output.txt','w+')

# Predict total sale of products for each day (for next 29 days) using forecast method of ARIMA
foreCast = results_ARIMA.forecast(steps = 29)
#plt.plot(ts,label='Train Data')
#plt.plot(foreCast, color='red',label='Prediction for 29 days' )
#plt.legend()
#plt.show()


#Write first prediction to  file 
w = '0\t'
for i in foreCast:
    i = int(round(i))
    w += str(i) + '\t'
file.write(w)
file.write('\n\n')

header_count = 0

# Test the Model before prediction
#trainModel_2(productData)

print("Predicting sale for each product for next 29 days...")

# Predict the sale for each product for each day ( for next 29 days)
for col in productData:
    parray = np.array(productData[col]).astype(float)
    pseries = pd.Series(parray)
    pseries.index = pd.to_datetime(pseries.index,unit = 'D')
    minAic=0
    orderparams=''
    seasonalparams=''
    for product_param in pdq:
        for product_param_seasonal in seasonal_pdq:
            aicmod = sm.tsa.statespace.SARIMAX(pseries,
                                                order=product_param,
                                                seasonal_order=product_param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

            product_results = aicmod.fit()
            if minAic==0:
                minAic=product_results.aic
                orderparams=product_param
                seasonalparams=product_param_seasonal
            if product_results.aic<minAic:
                minAic=product_results.aic
                orderparams=product_param
                seasonalparams=product_param_seasonal
                #print(minAic,orderparams,seasonalparams)
   
   
                # Apply SARIMAX for current product data
    parima = sm.tsa.statespace.SARIMAX(pseries,
                                        order=orderparams,
                                       seasonal_order=seasonalparams,
                                      enforce_stationarity=False,
                                         enforce_invertibility=False)
                
    results_ARIMA = parima.fit(disp=0)
    #model = sm.tsa.statespace.SARIMAX(nts, order=(1,1,1),seasonal_order=(0,1,1,7),enforce_stationarity=False,enforce_invertibility=False)
    #results_ARIMA = model.fit(disp=0)
    foreCast = results_ARIMA.forecast(steps = 29)
    s =str(productHeader[header_count]) + '\t'
    header_count += 1
    for i in foreCast:
        if i < 0:
            i = 0
        i = int(round(i))
        s += str(i) + '\t'
    file.write(s)
    file.write('\n\n')

file.close();
print("Prediction written successfully !")



