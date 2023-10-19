## Ambiguity Aversion Functions

import squigglepy as sq
from squigglepy.numbers import K, M, B
import numpy as np
import pandas as pd
import os

def cubic_weighting(i, N, coef):
    '''
    For a single index (representing percentile), it applies a cubic weighting function 
    to assign adjustments to the probabilities of each value in the sorted vector of payoffs.
    
    The coefficient will differ based on how ambiguity averse we are.
    '''
    w_i = (-1*coef*(i/(N-1)-1/2)**3+1)/(N)
    return w_i


def get_adjusted_probabilities(N, coef, weighting_function): 
    '''
    For all indices (representing percentiles), it applies the specified weighting
        function to the probabilities of each value in the sorted vector of payoffs.
    '''
    weights = np.zeros(N)
    for i in range(N):
        w_i = weighting_function(i, N, coef)
        weights[i] = w_i
    return weights

def check_sum(weights):
    '''
    Make sure they sum to nearly one (some error because there are an even number of sims, 
        as well as computational error)
    '''
    return np.sum(weights)

def sort_outcomes(x):
    '''
    Sort the outcomes in ascending order.
    '''
    sorted_x = np.sort(x)
    return sorted_x

def get_ambiguity_weighted_utility(eus, coef, weighting_function = cubic_weighting, to_print=False):
    '''
    This is the main function for calculating ambiguity-averse expected utility. 

    It takes in a vector of expected utilities (one for each simulation),
        and returns the ambiguity-averse expected utility given the specified weighting function. 
    '''
    N = len(eus)
    sorted_eus = np.sort(eus)
    w = get_adjusted_probabilities(N, coef, weighting_function)
    aa_eu = np.dot(w, sorted_eus)
    if to_print: 
        print("Sorted EUs: {}".format(sorted_eus))
        print("Ambiguity-neutral average outcome: {}".format(np.mean(eus)), eus)
        print("Adjusted Probability Weights: {}".format(w))
        print("Check sum: {}".format(check_sum(w)))
        print("Ambiguity-averse average outcome, not probability weighted: {}".format(aa_eu))

    return aa_eu

def make_aaev_dataframe(eus_dict, causes, folder = 'results', path = 'rp_mw', to_print=False):
    '''
    Make a dataframe for the AAEV for all causes considered and all the 
        different ambiguity aversion functions. 

    It also returns the percentiles of the expected utilities for each cause.
    '''
    aaev_0 = []
    aaev_4 = []
    aaev_8 = []
    aaef_quad = []
    cols = ["No ambiguity aversion", "Cubic, 1.5x weight to worst", "Cubic, 2x weight to worst", \
            "1st-pct", "5th-pct", "25th-pct", "Median", "75th-pct", "95th-pct", "99th-pct"]
    idx = []

    one_pct = []
    fifth_pct = []
    twentyfifth_pct = []
    median = []
    seventyfifth_pct = []
    ninetyfifth_pct = []
    nintyninth_pct = []

    for cause in causes:
        eus = eus_dict[cause]
        idx.append(cause)
        aaev_0.append(np.mean(eus))
        aaev_4.append(get_ambiguity_weighted_utility(eus, 4, cubic_weighting, to_print=False))
        aaev_8.append(get_ambiguity_weighted_utility(eus, 8, cubic_weighting, to_print=False))
        one_pct.append(np.percentile(eus, 1))
        fifth_pct.append(np.percentile(eus, 5))
        twentyfifth_pct.append(np.percentile(eus, 25))
        median.append(np.percentile(eus, 50))
        seventyfifth_pct.append(np.percentile(eus, 75))
        ninetyfifth_pct.append(np.percentile(eus, 95))
        nintyninth_pct.append(np.percentile(eus, 99))
        

    aaev_df = pd.DataFrame(list(zip(aaev_0, aaev_4, aaev_8, one_pct, fifth_pct, \
                                    twentyfifth_pct, median, seventyfifth_pct, ninetyfifth_pct, nintyninth_pct)), columns=cols, index=idx)
    sorted_aaev_df = aaev_df.sort_values(by="No ambiguity aversion", ascending=False)

    aaev_df.to_csv(os.path.join(folder,'{}_aaev_lower_p_xrisk_df.csv'.format(path)))
    return sorted_aaev_df

