# import the relevant packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import datetime
from dateutil.relativedelta import relativedelta
#===========================================================================================#
# Read in the data from FRED and format it
today = datetime.date.today()
yrs = 5
start = today - relativedelta(years=yrs)
end = today
mos = [1, 3, 6] # month maturities
yrs = [1,2,3,5,7,10,30] # year maturities
fred_codes = ['DGS'+ str(i) + 'MO' for i in mos] + ['DGS' + str(i) for i in yrs] #Fred maturity codes
df = web.DataReader(fred_codes, 'fred', start, end)
df = df.dropna()
df.columns = [str(i) + 'mo' for i in mos] + [str(i) + 'yr' for i in yrs]
#===========================================================================================#
# visualize the data
df.plot(figsize=(12,8))
plt.ylabel("Interest Rate")
plt.legend(bbox_to_anchor=(1.0, 1.0), loc=2)
plt.show()
#===========================================================================================#
# Normalize the data and get the covariance matrix
def pca(data, k=3):
    '''
    Function to determine the principal components
    '''
    m,n = data.shape
    mu = np.mean(data, axis=0)
    data_norm = data - mu
    sigma = np.std(data_norm, axis=0, ddof=1)
    data_norm /= sigma
    cov_mat = (1/(m-1)) * data_norm.T @ data_norm
    # Calculate the principal components
    values, vectors = np.linalg.eig(cov_mat)
    explained_variance = values[:k] / sum(values)
    return vectors, explained_variance
    
def pca_model(data, vectors, k=3):
    '''
    Function to determine the yield curve using the PCA model
    '''
    # Calculate the variance explained by each component
    pca_data = data @ vectors
    eig_score = pca_data[:, :k] #eigenscores
    # inverse transformation to model the yield curve using the firsk k principal components
    model_data = eig_score @ np.linalg.inv(vectors)[:k, :]
    return pca_data, model_data
#==========================================================================================#

# Measure of fit against the real yield curve
data = df.to_numpy()
k = 3
vectors, exp_var = pca(data, k=k)
pca_data, model_data = pca_model(data, vectors, k=k)
print('\nThe % variance explained by the k principal components are:', \
      *np.round(exp_var*100, 2))
#==========================================================================================#
    
# plot loadings for principal components
loadings = pd.DataFrame(vectors, columns=['PC'+str(i) for i in range(1,data.shape[1]+1)],\
                        index=df.columns)
loadings.plot()
plt.legend(bbox_to_anchor=(1.01, 0.9), loc=2)
plt.xlabel('Maturity Date')
plt.ylabel('Loadings')
plt.show()

# plot the time-series of first k factors
factors = pd.DataFrame(pca_data[:,:k], index=df.index, \
                       columns=['Factor' + str(i) for i in range(1,k+1)])
factors.plot()
#==========================================================================================#

# visualizing the Goodness of Fit 
for i in range(len(df.columns)):
    rmse = np.sqrt(np.mean(np.square(data[:,i] - model_data[:,i])))
    plt.plot(df.index, data[:,i], label='real')
    plt.plot(df.index, model_data[:,i], label='PCA model')
    plt.xlabel('DATE')
    plt.ylabel(df.columns[i] + ' rate')
    plt.title('RMSE = '+ str(round(rmse, 2)))
    plt.legend()
    plt.gca().yaxis.grid(True)
    plt.show()
#==========================================================================================#

# out of sample fit of the PCA model
n = 252  # len of test data
train_data = data[:-n]
test_data = data[-n:]
vectors, _ = pca(train_data, k=k)
model_data = pca_model(test_data, vectors, k=k)
#==========================================================================================#

for i in range(len(df.columns)):
    rmse = np.sqrt(np.mean(np.square(test_data[:,i] - model_data[:,i])))
    plt.plot(df.index, data[:,i], label='real')
    plt.plot(df.index[-n:], model_data[:,i], label='PCA model')
    plt.xlabel('DATE')
    plt.ylabel(df.columns[i] + ' rate')
    plt.title('RMSE = '+ str(round(rmse, 2)))
    plt.legend()
    plt.gca().yaxis.grid(True)
    plt.show()
#===========================================================================================#
