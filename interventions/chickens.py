### The file that simulates the cost-effectiveness of laying hen corporate campaigns (vector)
### The cost-effectiveness estimates are adjusted for p(sentience) and welfare ranges
###     based on the Rethink Priorities Moral Weight Project

import numpy as np
import pandas as pd
import squigglepy as sq
from squigglepy.numbers import K, M, B

N = 6*M

def sample_sc_human_eq_dalys_per_1000_chicken_campaign(moral_weight_lims=None, to_print=False): 
    '''
    Chicken-DALYs averted per $1000 spent on corporate campaigns
        From https://my.causal.app/models/143725/edit
        Conditional on being moral patients (sentient)
    '''
    lower_dalys_per_dollar = 0.52
    upper_dalys_per_dollar = 10.3

    chicken_dalys_per_dollar = sq.sample(sq.mixture([sq.lognorm(lower_dalys_per_dollar, upper_dalys_per_dollar), \
                                                     -0.5*sq.lognorm(lower_dalys_per_dollar, upper_dalys_per_dollar)], 
                                                     [0.97, 0.03]), N)

    chicken_dalys_per_1000 = chicken_dalys_per_dollar * 1000

    if moral_weight_lims is None:
        sc_moral_weights = chicken_sentience_conditioned_welfare_range()
    else: 
        lower = moral_weight_lims[0]
        upper = moral_weight_lims[1]
        sc_moral_weights = sq.sample(sq.uniform(lower, upper), N)
    
    sc_human_eq_chicken_dalys_per_1000 = chicken_dalys_per_1000 * sc_moral_weights
    if to_print:    
        print(f'Mean sentience-conditioned Chicken-DALYs per $1000: {np.mean(chicken_dalys_per_1000)}')
        print("Percentiles:")
        print(sq.get_percentiles(chicken_dalys_per_1000))
        print("Mean sentience-conditioned human-equivalent DALYs per $1000: {}".format(np.mean(sc_human_eq_chicken_dalys_per_1000)))
        print("Percentiles:")
        print(sq.get_percentiles(sc_human_eq_chicken_dalys_per_1000))

    return sc_human_eq_chicken_dalys_per_1000

def sample_is_chicken_sentient():
    '''
    Create a vector of binary variables representing whether a chicken is sentient,
        based on our subjective 90% CI on the probability that chickens are sentient.
    '''
    p_not_sent_low = 0.05
    p_not_sent_high = 0.25
    p_not_sent_lclip = 0.01
    p_not_sent_rclip = 0.5
    
    p_sent = np.ones(N) - sq.sample(sq.lognorm(p_not_sent_low, p_not_sent_high, lclip=p_not_sent_lclip, rclip=p_not_sent_rclip), N)
    binaries_is_sent = np.zeros(N)

    for i in range(N):
        X = np.random.binomial(1, p_sent[i])
        binaries_is_sent[i] = X

    return binaries_is_sent

def chicken_sentience_conditioned_welfare_range(to_print=False):
    '''
    Create a vector of welfare ranges for chickens, conditional on being sentient.
        Based on RP's moral weight project:
        https://docs.google.com/spreadsheets/d/1gJZlOTmrWwR6C7us5G0-aRM9miFeEcP11_6HEfpCPus/edit?usp=sharing
    '''
    wr_chicken_lower = 0.10 
    wr_chicken_upper = 0.974
    wr_chicken_lclip = 0.0024 # neuron count for a chicken
    wr_chicken_rclip = 1.282 # equal to the 95th percentile for the undiluted experiences model

    wr_chicken = sq.sample(sq.lognorm(wr_chicken_lower, wr_chicken_upper, lclip=wr_chicken_lclip, rclip=wr_chicken_rclip), N)

    if to_print:
        print(f'Mean welfare range for chickens: {np.mean(wr_chicken)}')
        print("Percentiles:")
        print(sq.get_percentiles(wr_chicken))

    return wr_chicken

def chicken_campaign_human_dalys_per_1000(to_print=False, moral_weight_range=None):
    '''
    Combine the sentience-conditioned chicken-DALYs per $1000 with the probability of sentience and the welfare range
        to get human-equivalent DALYs/$1000 for hen welfare campaigns. 
    The result is a vector of cost-effectiveness estimates. 
    '''
    if moral_weight_range is not None:
        sc_human_dalys_per_1000 = sample_sc_human_eq_dalys_per_1000_chicken_campaign(moral_weight_range)
    else: 
        sc_human_dalys_per_1000 = sample_sc_human_eq_dalys_per_1000_chicken_campaign(None)
    
    is_sent = sample_is_chicken_sentient()

    human_daly_equivalent_dalys_per_1000_chicken_campaign = sc_human_dalys_per_1000 * is_sent 

    human_daly_equivalent_dalys_per_1000_chicken_campaign = sc_human_dalys_per_1000 * is_sent

    if to_print:
        print(f'Mean human-DALYs per 1000 chickens: {np.mean(human_daly_equivalent_dalys_per_1000_chicken_campaign)}')
        print("Percentiles:")
        print(sq.get_percentiles(human_daly_equivalent_dalys_per_1000_chicken_campaign))

    return human_daly_equivalent_dalys_per_1000_chicken_campaign
