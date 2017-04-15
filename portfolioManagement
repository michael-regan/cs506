import csv
import scipy.io as sio
import numpy as np

# Implementing portfolio management to optimize performance of S&P stock investments using MWU; 
# data download from http://ocobook.cs.princeton.edu/links.htm

mat_contents = sio.loadmat('data_490_1000.mat')

## Checking basic functionality of arrays ##

# print(mat_contents['A'].shape)  # (490, 1000): 490 stocks, 1000 trading days

# print(mat_contents['A'].T[0])  # this is the column vector, price of all stocks on day 0: c_i^{0}


def to_csv(title, myArray):
    
	path='plots/plot_'+title+'.csv'

	with open(path, 'w') as g:
	    
	    writer=csv.writer(g)
	    writer.writerow(('day', 'value'))
	    for i in myArray:
	        writer.writerow(i)


## parameters ##

total_wealth_0=100000   # wealth on day 0

num_stocks = mat_contents['A'].shape[0] 

num_days = mat_contents['A'].shape[1] 

"""
Checking first investment strategy: Basic strategy (invest in all stocks equally, No changes; Let it ride)
What is value of portfolio at the close of each day?
"""

P_i_0 = 1/num_stocks  # fraction of your wealth invested in stock i on day 0

total_wealth_t=total_wealth_0  #initializing wealth for day t

myWealth=[(0, total_wealth_0)]

for t in range(0, num_days-1):

	r_i_t = mat_contents['A'].T[t+1]/mat_contents['A'].T[t]

	factor_wealth_increases_day_t = np.sum([x for x in P_i_0 * r_i_t])  # multiplicative factor wealth increases by on day t
	total_wealth_t = total_wealth_t * factor_wealth_increases_day_t
	#print("Day:", t+1, "Wealth:", total_wealth_t)
	myWealth.append((t+1, total_wealth_t))

to_csv('equal_noExperts', myWealth)
