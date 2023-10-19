## Weighted linear utility function

import numpy as np
import pandas as pd
import squigglepy as sq
from squigglepy.numbers import K, M, B 

def weight_function_aggressive(x, power = 0.25, to_print = False): 
    '''
    Given a vector of "payoffs" from an empirical distribution, 
        calculate the weight of each payoff. 
    '''
    w_x = []
    for x_i in x:
        if x_i < 0:
            w_i = np.log(1 - x_i) + 1
        else:
            w_i = 1/(1 + x_i**(power))
        w_x.append(w_i)
    w_x = np.array(w_x)

    if to_print:
        for i in range(len(x)):
            print(f'Weight for payoff {x[i]} is {w_x[i]}')

    return w_x

def weight_function_symmetric(x, power = 0.25, to_print = False): 
    '''
    Given a vector of "payoffs" from an empirical distribution, 
        calculate the weight of each payoff. 
    '''
    w_x = []
    for x_i in x:
        if power == 0:
            w_i = 1
        else:
            if x_i < 0:
                w_i = 2 - 1/(1 + (-1*x_i)**(power))
            else:
                w_i = 1/(1 + x_i**(power))
        w_x.append(w_i)
    w_x = np.array(w_x)

    if to_print:
        for i in range(len(x)):
            print(f'Weight for payoff {x[i]} is {w_x[i]}')

    return w_x

def weight_function_max_1(x, power = 0.25, to_print = False):
    '''
        Given a vector of "payoffs" from an empirical distribution, 
        calculate the weight of each payoff. 
    '''
    w_x = []
    for x_i in x:
        if x_i < 0:
            w_i = 1
        else:
            w_i = 1/(1 + x_i**(power))
        w_x.append(w_i)
    w_x = np.array(w_x)

    if to_print:
        for i in range(len(x)):
            print(f'Weight for payoff {x[i]} is {w_x[i]}')

    return w_x

def get_probability(x, p):
    '''
    Given a vector of "payoffs" from an empirical distribution, 
        calculate the probability of each payoff. 
    '''
    if len(p) > 0:
        p_x = np.array(p)
    else:
        N = len(x)
        p_x = np.array([1/N]*N)   

    return p_x

def get_average_weight(x, weight_function=weight_function_symmetric, power=0.25, p=[], to_print = False):
    '''
    Given a vector of "payoffs" from an empirical distribution, 
        calculate the average weight of the payoffs. 
    '''
    w_x = weight_function(x, power)
    p_x = get_probability(x, p)
    avg_w = np.sum(w_x * p_x)

    if to_print:
        print(f'Average weight is {avg_w}')

    return avg_w

def get_coefficients(x, weight_function=weight_function_symmetric, power=0.25, p=[], to_print = False):
    '''
    Given a vector of "payoffs" from an empirical distribution, 
        calculate the coefficients of the linear utility function. 
    '''
    w_x = weight_function(x, power)
    avg_w = get_average_weight(x, weight_function, power, p)

    c_x = w_x / avg_w

    if to_print:
        for i in range(len(x)):
            print(f'Coefficient for payoff {x[i]} is {c_x[i]}')
    
    return c_x

def get_weighted_utility_each_outcome(x, weight_function =weight_function_symmetric, power=0.25, p=[], to_print = False):
    '''
    Given a vector of "payoffs" from an empirical distribution, 
        calculate the weighted utility of each payoff (vector). 
    '''
    c_x = get_coefficients(x, weight_function, power, p)
    p_x = get_probability(x, p)
    
    wu_x = c_x*p_x*x
    if to_print:
        for i in range(len(x)):
            print(f'Utility for payoff {x[i]} is {wu_x[i]}')

    return wu_x

def get_weighted_linear_utility(x, power = 0.25, weight_function =weight_function_symmetric, p=[], to_print = False):
    '''
    Given a vector of "payoffs" from an empirical distribution, 
        calculate the weighted linear utility (scalar). 
    '''
    wu_x = get_weighted_utility_each_outcome(x, weight_function, power, p)
    wlu = np.sum(wu_x)

    if to_print:
        print(f'Weighted linear utility is {wlu}')

    return wlu

def get_table(x , weight_function = weight_function_symmetric, power=0.25, p=[]):
    '''
    Given a vector of "payoffs" from an empirical distribution and a WLU function, 
        calculate the weighted utility of each payoff (vector). 
    '''
    w_x = weight_function(x, power)
    c_x = get_coefficients(x, weight_function, power, p)
    avg_wt = get_average_weight(x, weight_function, power, p)
    p_x = get_probability(x, p)
    wu_x = get_weighted_utility_each_outcome(x, weight_function, power, p)
    wlu = get_weighted_linear_utility(x, power, weight_function, p)

    print("Expected Value: {}".format(np.round(np.mean(x),2)))
    print("Weighted Linear Utility: {}".format(np.round(wlu,2)))
    print("Average Weight: {}".format(np.round(avg_wt,4)))
    print("Power: {}".format(power))
    df = pd.DataFrame({'Payoff': x, 
                       'Weight': w_x, 
                       'Probability': p_x,
                       'Coefficient': c_x, 
                       'Contribution to Weighted Utility': wu_x,
                       })
    print("Check: {}".format(np.round(np.sum(wu_x),2)))
    print(df)

    return df




