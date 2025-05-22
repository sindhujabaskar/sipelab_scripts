import numpy as np
import pandas as pd
import plotly.express as px
import scipy.optimize as opt

# Read the BEC data from the file
standards_csv = pd.read_csv(r'D:\Projects\BEC_quantification\25-04-18 14-10-04 250418_BEC_run02_SB_standard curve.csv', encoding = "ISO-8859-1")
samples_csv = pd.read_csv(r"D:\Projects\BEC_quantification\25-04-18 14-10-04 250418_BEC_run02_SB_samples.csv", encoding = "ISO-8859-1")

#Index the data to just obtain the relevant columns
standard_values = standards_csv.loc[18:]
sample_values = samples_csv.loc[18:]

#add a column with all standard concentrations
standard_values['std_conc'] = ['std_conc',98,98,98,87,87,87,66,66,66,44,44,44,22,22,22,11,11,11,0.1,0.1,0.1,0.1,0.1,0.1]

#rename columns
standard_values.columns = standard_values.iloc[0]
sample_values.columns = sample_values.iloc[0]

#Plot the data and label the graph
x = np.asarray(standard_values['std_conc'][1:])
y = np.asarray(standard_values['Raw Data  (595)'][1:])
px.scatter(x = x, y = y, log_x = True, range_y = [25,0], title = 'BEC Standards')

# Linear fit function 
def fit_func(x, m, b):
    return m * x + b

# 4-parameter logistic curve function
def func(xdata, a, b, c, d): 
    return ((a-d)/(1.0+((xdata/c)**b))) + d

# use scipy's curve_fit to obtain the parameters and covariance of the fit
a_fit, cov = opt.curve_fit(func, x, y)

#plot the fitted curve
px.line(x=x, y= func(x, *a_fit), log_x= True)

#next steps:
# Plot the unknown samples on the fitted curve 
# Use the fitted curve to calculate the concentration of the unknown samples

