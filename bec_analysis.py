#%%
import pandas as pd
import numpy as np
import scipy

# LOAD DATA
BEC_samples = pd.read_csv(r"C:\Users\sturt\OneDrive - The Pennsylvania State University\Sipe Lab\Projects\BEC Quantification\250703_BEC.csv")
print(BEC_samples.head())

BEC_standards = pd.read_csv(r"C:\Users\sturt\OneDrive - The Pennsylvania State University\Sipe Lab\Projects\BEC Quantification\250703_BEC_standard.csv")
print(BEC_standards.head())

# DEFINE FUNCTIONS
def bec_avg(df):
    avg_OD = df.groupby(['Group','Sample'])['OD'].mean().reset_index()
    avg_OD = avg_OD.rename(columns={'OD': 'OD_avg'})
    return avg_OD

bec = BEC_samples[['Group','Sample','OD']]

BEC_standards.rename(columns= {'Well':'Well','Sample':'Sample','Concentration (mM)':'Conc','OD':'OD'}, inplace=True)
std_log = np.log10(BEC_standards.OD)
linregress = scipy.stats.linregress(x= BEC_standards.Conc, y= std_log)

bec_avg = bec_avg(bec)
print(bec_avg.head())


# Create the linear regression equation function
def calculate_concentration(od_value, slope, intercept):
    """
    Calculate concentration from OD value using the linear regression equation.
    """
    log_od = np.log10(od_value)
    concentration = (log_od - intercept) / slope
    return concentration

# Extract the regression parameters
slope = linregress.slope
intercept = linregress.intercept

# Function to calculate concentrations for your samples
def predict_concentrations(od_values):
    """
    Predict concentrations from OD values using the calibration curve.
    """
    concentrations = []
    for od in od_values:
        if od > 0:  # Can't take log of negative or zero values
            conc = calculate_concentration(od, slope, intercept)
            concentrations.append(conc)
        else:
            concentrations.append(np.nan)
    return concentrations

# Apply to your sample data
bec_avg['Predicted_Conc'] = predict_concentrations(bec_avg['OD_avg'])

# Merge with original data to get EtOH dose and Time columns
bec_with_metadata = bec_avg.merge(BEC_samples[['Group', 'Sample', 'EtOH dose', 'Time']].drop_duplicates(), 
                                  on=['Group', 'Sample'], how='left')

# Group by EtOH dose and Time, calculate mean and standard error
dose_time_analysis = bec_with_metadata.groupby(['EtOH dose', 'Time'])['Predicted_Conc'].agg([
    'mean', 
    'std', 
    'count',
    lambda x: x.std() / np.sqrt(x.count())  # Standard error
]).reset_index()

# Rename the lambda column to 'sem'
dose_time_analysis.columns = ['EtOH dose', 'Time', 'Mean_Conc', 'Std_Conc', 'Count', 'SEM_Conc']

print("\nBEC Analysis by Dose and Time:")
print(dose_time_analysis)

# Filter for specific conditions if needed
filtered_analysis = dose_time_analysis[
    (dose_time_analysis['EtOH dose'].isin(['1g/kg', '2g/kg'])) & 
    (dose_time_analysis['Time'].isin([15, 45]))
]

print("\nFiltered Analysis (1g/kg and 2g/kg at 15 and 45 min):")
print(filtered_analysis)

















# import numpy as np
# import pandas as pd
# import plotly.express as px
# import scipy.optimize as opt

# # Read the BEC data from the file
# standards_csv = pd.read_csv(r'D:\Projects\BEC_quantification\25-04-18 14-10-04 250418_BEC_run02_SB_standard curve.csv', encoding = "ISO-8859-1")
# samples_csv = pd.read_csv(r"D:\Projects\BEC_quantification\25-04-18 14-10-04 250418_BEC_run02_SB_samples.csv", encoding = "ISO-8859-1")

# #Index the data to just obtain the relevant columns
# standard_values = standards_csv.loc[18:]
# sample_values = samples_csv.loc[18:]

# #add a column with all standard concentrations
# standard_values['std_conc'] = ['std_conc',98,98,98,87,87,87,66,66,66,44,44,44,22,22,22,11,11,11,0.1,0.1,0.1,0.1,0.1,0.1]

# #rename columns
# standard_values.columns = standard_values.iloc[0]
# sample_values.columns = sample_values.iloc[0]

# #Plot the data and label the graph
# x = np.asarray(standard_values['std_conc'][1:])
# y = np.asarray(standard_values['Raw Data  (595)'][1:])
# px.scatter(x = x, y = y, log_x = True, range_y = [25,0], title = 'BEC Standards')

# # Linear fit function 
# def fit_func(x, m, b):
#     return m * x + b

# # 4-parameter logistic curve function
# def func(xdata, a, b, c, d): 
#     return ((a-d)/(1.0+((xdata/c)**b))) + d

# # use scipy's curve_fit to obtain the parameters and covariance of the fit
# a_fit, cov = opt.curve_fit(func, x, y)

# #plot the fitted curve
# px.line(x=x, y= func(x, *a_fit), log_x= True)

#next steps:
# Plot the unknown samples on the fitted curve 
# Use the fitted curve to calculate the concentration of the unknown samples

