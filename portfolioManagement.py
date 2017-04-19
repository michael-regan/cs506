import csv, math
import scipy.io as sio
import numpy as np

# Implementing portfolio management to optimize performance of S&P stock investments using multiplicative weights update (MWU); 
# Based on lecture notes from Arora, 2016 (chapter 10)
# data from http://ocobook.cs.princeton.edu/links.htm

mat_contents = sio.loadmat('data_490_1000.mat')

## Checking basic functionality of arrays ##

# print(mat_contents['A'].shape)  # (490, 1000): 490 stocks, 1000 trading days

# print(mat_contents['A'].T[0])  # this is the column vector, price of all stocks on day 0: c_i^{0}


"""Parameters"""

total_wealth_0=100000   # wealth on day 0

num_stocks = mat_contents['A'].shape[0] 

num_days = mat_contents['A'].shape[1] 


## end parameters ##


def to_csv(title, myArray):
    
	path='plots/plot_'+title+'.csv'

	with open(path, 'w') as g:
	    
	    writer=csv.writer(g)
	    writer.writerow(('day', 'value'))
	    for i in myArray:
	        writer.writerow(i)


def MWU(distribution_experts, payoff, learning_rate):
    
    mwu = [1+x*learning_rate for x in payoff]
  
    new_distribution = [x*y for x,y in zip(distribution_experts,mwu)]

    normalized_new_dist = [x/sum(new_distribution) for x in new_distribution]
    
    return normalized_new_dist
    
    
"""
Basic investment strategy: Invest in all stocks equally, No changes --> Let it ride;
What is value of portfolio at the end of 1000 days of trading?
"""

P_i_0 = 1/num_stocks  # fraction of your wealth invested in stock i on day 0 (this proportion does not change in this first test)

total_wealth_t=total_wealth_0  #initializing wealth for day t

myWealth=[(0, total_wealth_0)]

for t in range(0, num_days-1):

	r_i_t = mat_contents['A'].T[t+1]/mat_contents['A'].T[t]

    # multiplicative factor wealth increases by on day t with EQUAL proportion of wealth invested in each stock
	factor_wealth_increases_day_t = np.sum([x for x in P_i_0 * r_i_t])  
	total_wealth_t = total_wealth_t * factor_wealth_increases_day_t
	#print("Day:", t+1, "Wealth:", total_wealth_t)
	myWealth.append((t+1, total_wealth_t))

#to_csv('equal_noExperts', myWealth)


"""
Next strategy: P_i_t is the fraction of wealth invested in stock i at the start of day t.
The distribution on experts is the way of splitting our money into the n stocks.
Maximize payoffs: Increase the weight of experts if they get positive payoff, and reduce weight in case of negative payoff.
"""

learning_rates = [1.5]

for learning_rate in learning_rates:

    learning_rates.append(learning_rate+0.1)

    if learning_rate>10:
        break

    print (learning_rate)

    P_i_t = [1/num_stocks]*num_stocks  # fraction of your wealth invested in stock i on day t (initialization; this will be updated according to payoffs)

    total_wealth_t=total_wealth_0  #initializing wealth for day t

    myWealth=[(0, total_wealth_0)]

    for t in range(0, num_days-1):

        r_i_t = mat_contents['A'].T[t+1]/mat_contents['A'].T[t]

        # payoff for expert i on day t
        payoff_day_t = [math.log(x) for x in r_i_t]
        
        # new proportion for each expert on day t
        P_i_next_t = MWU(P_i_t, payoff_day_t, learning_rate)

        P_i_t=P_i_next_t  # updating for next iteration
        
        #print(P_i_next_t)
        
        # multiplicative factor wealth increases by on day t, now with proportion of wealth investing being updated 
        factor_wealth_increases_day_t = np.sum([x for x in P_i_next_t * r_i_t])  
        total_wealth_t = total_wealth_t * factor_wealth_increases_day_t
        
        if (t+1)%100==0:
            print("Day:", t+1, "Wealth:", total_wealth_t)
        myWealth.append((t+1, total_wealth_t))

    print()

    #to_csv('maximizePayoff_5', myWealth)
