import numpy as np
import pandas
import statsmodels.api as sm

"""
In this question, you need to:
1) implement the linear_regression() procedure
2) Select features (in the predictions procedure) and make predictions.

"""

def linear_regression(features, values):
    """
    Perform linear regression given a data set with an arbitrary number of features.
    
    This can be the same code as in the lesson #3 exercise.
    """
    
    ###########################
    ### YOUR CODE GOES HERE ###
    ###########################
    features = sm.add_constant(features)
    model = sm.OLS(values, features)
    results = model.fit()
    intercept = results.params[0]
    params = results.params[1:]
    
    return intercept, params

def predictions(dataframe, features = ['rain', 'precipi', 'hour', 'tempi'], target = 'ENTRIESn_hourly'):
    '''
    The NYC turnstile data is stored in a pandas dataframe called weather_turnstile.
    Using the information stored in the dataframe, let's predict the ridership of
    the NYC subway using linear regression with gradient descent.
    
    You can download the complete turnstile weather dataframe here:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv    
    
    Your prediction should have a R^2 value of 0.40 or better.
    You need to experiment using various input features contained in the dataframe. 
    We recommend that you don't use the EXITSn_hourly feature as an input to the 
    linear model because we cannot use it as a predictor: we cannot use exits 
    counts as a way to predict entry counts. 
    
    Note: Due to the memory and CPU limitation of our Amazon EC2 instance, we will
    give you a random subet (~10%) of the data contained in 
    turnstile_data_master_with_weather.csv. You are encouraged to experiment with 
    this exercise on your own computer, locally. If you do, you may want to complete Exercise
    8 using gradient descent, or limit your number of features to 10 or so, since ordinary
    least squares can be very slow for a large number of features.
    
    If you receive a "server has encountered an error" message, that means you are 
    hitting the 30-second limit that's placed on running your program. Try using a
    smaller number of features.
    '''
    # Select Features (try different features!)
    features = dataframe[features]
    
    # Add UNIT to features using dummy variables
    dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')
    features = features.join(dummy_units)
    
    # Values
    values = dataframe[target]
    
    # Get the numpy arrays
    features_array = features.values
    values_array = values.values

    # Perform linear regression
    intercept, params = linear_regression(features_array, values_array)
    
    predictions = intercept + np.dot(features_array, params)
    return predictions

def compute_r_squared(data, predictions):
    '''
    In exercise 5, we calculated the R^2 value for you. But why don't you try and
    and calculate the R^2 value yourself.
    
    Given a list of original data points, and also a list of predicted data points,
    write a function that will compute and return the coefficient of determination (R^2)
    for this data.  numpy.mean() and numpy.sum() might both be useful here, but
    not necessary.

    Documentation about numpy.mean() and numpy.sum() below:
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html
    '''
    
    # your code here
    r_squared = 1 - ((data-predictions)**2).sum()/((data - data.mean())**2).sum()
    
    return r_squared

f = ['rain', 'precipi', 'hour', 'meantempi']
t = 'ENTRIESn_hourly'
df = pandas.read_csv('4.1-visualization/turnstile_weather_v2.csv')
pred = predictions(df, f, t)

dft = df[t]
rs = compute_r_squared(dft, pred)

print('r-squared: ', rs)

### Plotting Residuals

import matplotlib.pyplot as plt

def plot_residuals(actual, predictions):
    '''
    Using the same methods that we used to plot a histogram of entries
    per hour for our data, why don't you make a histogram of the residuals
    (that is, the difference between the original hourly entry data and the predicted values).
    Try different binwidths for your histogram.

    Based on this residual histogram, do you have any insight into how our model
    performed?  Reading a bit on this webpage might be useful:

    http://www.itl.nist.gov/div898/handbook/pri/section2/pri24.htm
    '''
    
    plt.figure()
    (actual - predictions).hist(bins=100)
    plt.title("Total Residuals")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    return plt

plot_residuals(dft, pred)
plt.savefig('residuals.png')
plt.clf()
print("Created plot: residuals.png")

### Plotting Residuals per Data Point
# import pdb
# pdb.set_trace()
plt.plot(range(1, len(pred)+1), dft - pred)
plt.title("Residuals Per Data Point")
plt.xlabel("nth Data Point")
plt.ylabel("Residuals")
plt.axhline()
plt.savefig('residuals_per_data_point.png')
plt.clf()

plt.scatter(range(1, 1001), (dft - pred)[:1000])
plt.title("Residuals Per Data Point (first 1000 points)")
plt.xlabel("nth Data Point")
plt.ylabel("Residuals")
plt.axhline()
plt.savefig('residuals_per_data_point-zoomed.png')
plt.clf()

print("Created plot: residuals_per_data_point-zoomed.png")
