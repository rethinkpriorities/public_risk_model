### Functions relating to the implementation of Difference-Making Risk-Weighted Expected Utility (DMREU) ###
### inspired by Lara Buchak's paper, "RISK AND TRADEOFFS" at https://philpapers.org/archive/BUCRAT.pdf

import numpy as np
import pandas as pd
import squigglepy as sq

def sort_payoffs(x):
    '''
    Given a vector of "payoffs" from an empirical distribution, 
        sort the vector in ascending order.
    '''
    x = np.array(x)
    sorted_x = np.sort(x)
    return sorted_x

def get_probability_at_least_xi(sorted_x, p = []):
    '''
    Given a sorted vector of "payoffs" from an empirical distribution, 
        calculate the probability that the payoff is at least as big as each value (vector). 
    This is determined by the index of the payoff in the sorted vector.
    '''
    N = len(sorted_x)
    P = np.zeros(N)
  
    for i in range(N):
        if len(p) == 0:
            P[i] = 1-i/N
        else:
            P[i] = np.sum(p[i:])
    P = np.array(P)
    return P

def risk_function(P, a = 2): 
    '''
    Given a vector of probabilities (of getting at least a certain payoff), 
        apply a risk function to each probability. Default is raising to the second power
    Returns a vector of risk-weighted probabilities. 
    '''
    r_P = P**a

    return r_P

def one_value_dmreu_contributions(i, sorted_x, r_P):
    '''
    DMREU for one value from the payoffs vector. 
    '''
    U_x_i = sorted_x[i] # assuming linear utility function
    r_P_i = r_P[i]
    dmreu_contribution_i = 0
    if i > 0:
        U_x_i_minus_1 = sorted_x[i-1]
        dmreu_contribution_i = r_P_i * (U_x_i - U_x_i_minus_1)
    else:
        dmreu_contribution_i = r_P_i * U_x_i

    return dmreu_contribution_i

def create_table(sorted_x, P, r_P, dmreu_contributions_x):
    '''
    Given a sorted vector of "payoffs" from an empirical distribution, 
        calculate the probability that the payoff is at least as big as each value (vector). 
    This is determined by the index of the payoff in the sorted vector.
    '''
    difference_from_next_best = np.zeros(len(sorted_x))
    difference_utility_from_next_worst = np.zeros(len(sorted_x))
    for i in range(len(sorted_x)-1):
        difference_from_next_best[i] = r_P[i] - r_P[i+1]
    for i in range(1, len(sorted_x)):
        difference_utility_from_next_worst[i] = sorted_x[i] - sorted_x[i-1]

    
    table = pd.DataFrame({'Payoff': sorted_x, 'Probability of getting at least X': np.round(P, 5), 'Risk-Weighted Probability': np.round(r_P, 5), \
                          'Difference in Risk-Weighted Prob. from Next Outcome': np.round(difference_from_next_best, 6), \
                          'Difference in Utility from Next Worst Outcome': np.round(difference_utility_from_next_worst, 5), \
                          'DMREU contribution': np.round(dmreu_contributions_x, 5)})
    return table

def get_risk_weighted_utility(x, a, p=[], to_print = False):
    '''
    Given a vector of "payoffs" from an empirical distribution, 
        calculate the difference-making risk-averse expected value of each payoff (vector). 
    '''
    N = len(x)
    dmreu_contributions_x = np.zeros(N)
    sorted_x = sort_payoffs(x)
    P = get_probability_at_least_xi(sorted_x, p)
    r_P = risk_function(P, a)

    for i in range(N):
        dmreu_contributions_x[i] = one_value_dmreu_contributions(i, sorted_x, r_P)

    DMREU = round(np.sum(dmreu_contributions_x), 5)
    table = create_table(sorted_x, P, r_P, dmreu_contributions_x)

    if to_print:
        if p == []:
            print(f'Expected Value is {np.mean(x)} human-equivalent DALYs/$1000')
        else:
            print(f'Expected Value is {np.dot(x, p)} human-equivalent DALYs/$1000')
        print(f'Difference-making risk-weighted expected utility is {DMREU}')
        print(table)

    return DMREU


