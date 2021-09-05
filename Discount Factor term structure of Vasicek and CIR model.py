#=============================Import the relevant packages==================================#
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
#===========download data and format it==============#
today=datetime.date.today()
yrs=1
start=today-relativedelta(years=yrs)
end=today    
df=web.DataReader('DGS3MO','fred',start,end).rename(columns={'DGS3MO':'Rf'})\
    .fillna(method='ffill')#read in the 3-mo cons. mat. treasury rate
if(df.Rf.isnull()[0]):
    df.Rf[0]=df.Rf[1]
df['U1']=1
delta=1/252
#============================================VASCICEK=======================================#
# A Function to calculate the Vascicek model implied discount factor term structures (P(t,T))
def pvasicek(df, delta, t, T, r):    
    yy=np.array([df.Rf[1:len(df)]]).T
    xx=np.concatenate((np.array([df.U1[0:len(df)-1]]).T,\
                   np.array([df.Rf[0:len(df)-1]]).T),axis=1)
    beta=np.linalg.inv(xx.T@xx)@xx.T@yy    
    k=(1-beta[1][0])/delta    
    theta=beta[0][0]/(k*delta)
    err=yy-xx@beta
    sigma=err.std()/(np.sqrt(delta))    
    B = (1/k)*(1-np.exp(-k*(T-t)))
    A = np.exp(((theta - (sigma**2/(2*k**2))) * (B-T+t)) - ((sigma**2/(4*k))*(B**2)))
    P = A * np.exp(-B * r)
    return P
#============================================CIR============================================#
# A Function to calculate the CIR model implied discount factor term structures (P(t,T))
def pcir(df, delta, t, T, r):
    df = df[df.Rf > 0]    # to avoid division by 0   
    mat = np.concatenate((np.matrix(df.Rf[1:len(df)]).T,\
                   np.matrix(df.Rf[0:len(df)-1]).T),axis=1)
    yy = (mat[:,0]-mat[:,1])/np.sqrt(mat[:,1])
    xx = np.concatenate((1/np.sqrt(mat[:,1]),np.sqrt(mat[:,1])),axis=1)
    beta = np.linalg.inv(xx.T@xx)@xx.T@yy
    err = yy-xx@beta
    k = -beta[1,0]/delta
    theta = beta[0,0]/(k*delta)
    sigma = err.std()/np.sqrt(delta)
    h = np.sqrt(k**2 +(2*sigma**2))
    B_numer = 2*(np.exp(h*(T-t)) - 1) # numerator of B
    B_denom = A_denom = (2*h) + ((k+h)*(np.exp(h*(T-t))-1)) # denominator of A and B
    A_numer = (2*h)*np.exp((k+h)*(T-t)/2) # numerator of A
    A_pow = (2*k*theta)/(sigma**2) # Exponent term in A
    B = B_numer / B_denom
    A = (A_numer/A_denom)**A_pow
    P = A * np.exp(-B*r)
    return P
#=======================Plotting the Discount Factor Term Structure==========================
t = 0
rf = df.Rf.iloc[-1] / 100 # current risk free rate
Ts = [i for i in range(20)]
vas = [pvasicek(df,delta,t,T,rf) for T in Ts]
cir = [pcir(df,delta,t,T,rf) for T in Ts]

plt.plot(Ts, vas, label ='Vasicek Model')
plt.plot(Ts, cir, label= 'CIR Model')
plt.title("Term structure of discount factors"+" ("+str(today)+")")
plt.xlabel("Time to Maturity")
plt.ylabel("Discount factor")
plt.legend()
plt.show()

