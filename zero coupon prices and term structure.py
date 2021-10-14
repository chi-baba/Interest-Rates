# Import the relevant packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import dateutil
#=====================================================================================#
# filepaths of the data
# the data was gotten from https://www.wsj.com/market-data/bonds/treasuries
f1 = 'bills2021-10-08.txt'
f2 = 'bonds2021-10-08.txt'
# Read the treasury data and format it
bills = pd.read_csv(f1, sep='\t')
bonds = pd.read_csv(f2, sep='\t')
bills = bills.drop(bills.index[bills["ASKED"]=='n.a'])
bonds = bonds.drop(bonds.index[bonds["ASKED"]=='n.a'])
#=====================================================================================#
# Adjust the bond price from a fraction of 32
bonds['ASKED'] = bonds.ASKED//1  + (bonds.ASKED%1 * (100/32))
#=====================================================================================#
# convert the maturities to datetime
bonds['MATURITY'] = pd.to_datetime(bonds.MATURITY)
bills['MATURITY'] = pd.to_datetime(bills.MATURITY)
#=====================================================================================#
# Remove entries from the bonds table with maturities <= 1 year from today
today = pd.to_datetime(f1[-14 : -4]) # the date of the treasury data
bonds = bonds[bonds.MATURITY - today > datetime.timedelta(365)]
#=====================================================================================#
# drop duplicate treasury Maturity observations except the first
bonds = bonds.drop_duplicates(subset=['MATURITY'])
bills = bills.drop_duplicates(subset=['MATURITY'])
#=====================================================================================#
# Calculate time-to-maturity in years and determine bills price
bonds['TTM'] = np.round((bonds.MATURITY - today) / datetime.timedelta(365),6)
bills['TTM'] = np.round((bills.MATURITY - today) / datetime.timedelta(365),6)
# calculate Treasury bill prices
bills['PRICE'] = 1 / (1 + (bills['ASKED YIELD'] / 100) * bills.TTM)
#=====================================================================================#
# Set the quoted bond price
bonds['PRICE'] = bonds['ASKED'] / 100
bonds['ZERO PRICE'] = bonds['ASKED'] / 100
#=====================================================================================#
# reset the treasury data index
bills.index = [i for i in range(len(bills))]
bonds.index = [i for i in range(len(bonds))]
#======================================================================================#
# Bootstrap the zero-coupon bond prices (Semi-Annual coupon payments)
for i in range(len(bonds)): #Iterates over all the bonds
    c = bonds.loc[i, 'COUPON'] / 100
    n = int(np.floor(bonds.loc[i, 'TTM'])*2) # number of coupon payments to bootstrap
    prc = bonds.loc[i, 'ZERO PRICE']
    for j in range(n,0,-1):
        cpndate = bonds.loc[i, 'MATURITY'] - dateutil.relativedelta.relativedelta(months=(j*6))
        if (cpndate - today) / datetime.timedelta(365) < 1:
            diff = abs(bills.MATURITY - cpndate)
            p = bills.PRICE[diff.idxmin()]
        else:
            diff = abs(bonds.MATURITY - cpndate)
            p = bonds['ZERO PRICE'][diff.idxmin()]
        if j==n:
            # add accrued interest to the published "clean" price
            bonds.loc[i, 'ZERO PRICE'] = prc + (c/2)*(1 - (cpndate-today)/datetime.timedelta(30*6))
        bonds.loc[i, 'ZERO PRICE'] -= (c*p/2)
    bonds.loc[i, 'ZERO PRICE'] = bonds.loc[i, 'ZERO PRICE'] / ( 1 + (c/2) )
    # correct for numerical errors resulting in large jumps in the zerio yield
    if i > 0 and (bonds.loc[i, 'ZERO PRICE'] / bonds.loc[i-1, 'ZERO PRICE'] - 1) > 0.01:
        zp_prev = bonds.loc[i-1, 'ZERO PRICE']
        ttm = bonds.loc[i, 'TTM']
        ttm_prev = bonds.loc[i-1, 'TTM']
        bonds.loc[i, 'ZERO PRICE'] = 1/((1+1/(zp_prev**(1/ttm_prev))-1)**ttm)
#=====================================================================================#
# # Merge the treasury data
df = pd.DataFrame(columns = ['MATURITY','TTM','PRICE','ASKED YIELD','ZERO PRICE'])
df['MATURITY'] = (bills.MATURITY).append(bonds.MATURITY)
df['TTM'] = (bills.TTM).append(bonds.TTM)
df['PRICE'] = (bills.PRICE).append(bonds.PRICE)
df['ASKED YIELD'] = (bills['ASKED YIELD']).append(bonds['ASKED YIELD'])
df['ZERO PRICE'] = (bills.PRICE).append(bonds['ZERO PRICE'])
df = df.set_index('MATURITY')
#=====================================================================================#
# Plotting the term-structure of discount factors (zero prices)
df['ZERO PRICE'].plot(color='k')
plt.ylabel('discount factor')
plt.grid()
plt.title('Discount Factor term-structure ('+str(today)[:10]+')')
plt.show()
#=====================================================================================#
# Determine the zero-coupon bond yield
df['ZERO YIELD'] = ( ( 1 / ( df['ZERO PRICE']**(1/df.TTM) ) ) - 1 ) * 100
#=====================================================================================#
# Plotting the term-structure of interest rates (yield curve)
df['ZERO YIELD'].plot(color='b')
plt.ylabel('Yield')
plt.grid()
plt.title('term-structure of interest rates ('+str(today)[:10]+')')
plt.show()

