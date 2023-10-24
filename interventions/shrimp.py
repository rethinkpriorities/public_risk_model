### The file that simulates the cost-effectiveness of shrimp corporate campaigns (vector)
### The cost-effectiveness estimates are adjusted for p(sentience) and welfare ranges

import numpy as np
import pandas as pd
import squigglepy as sq
from squigglepy.numbers import K, M, B

N = 8*M

### GENERIC SHRIMP INTERVENTION 

example_shrimp_project_dict = {
    'harm_type': ['slaughter'],
    'low_suffering_reduced_proportion': {'slaughter': 0.5},
    'high_suffering_reduced_proportion': {'slaughter': 0.9},
    'low_p_success': 0.2,
    'high_p_success': 0.8,
    'low_p_shrimp_affected': 5*10**-5,
    'high_p_shrimp_affected': 5*10**-3,
    'low_cost': 100*K,
    'high_cost': 500*K,
    'low_years_credit': 1,
    'high_years_credit': 5,
    }

waterquality_shrimp_project = {
    'harm_types': ['ammonia', 'low_salinity', 'low_dissolved_oxygen', 'high_temp', 'low_temp', 'ph', 'water_pollution'],
    'low_suffering_reduced_proportion': {'ammonia': 0.3, 'low_salinity': 0.3, 'low_dissolved_oxygen': 0.3,
                                        'high_temp': 0.3, 'low_temp': 0.3, 'ph': 0.3, 'water_pollution': 0.3}, # low end on proportion of shrimp suffering reduced by intervention
    'high_suffering_reduced_proportion': {'ammonia': 0.7, 'low_salinity': 0.7, 'low_dissolved_oxygen': 0.7,
                                        'high_temp': 0.7, 'low_temp': 0.7, 'ph': 0.7, 'water_pollution': 0.7}, # high end on proportion of shrimp suffering reduced by intervention
    'low_p_success': 0.4, # low end on probability of intervention succeeding
    'high_p_success': 0.8, # high end on probability of intervention succeeding
    'low_p_shrimp_affected': 5*10**-4, # low end on proportion of shrimp in world affected by an individual campaign
    'high_p_shrimp_affected': 5*10**-3, # high end on proportion of shrimp being affected by an individual campaign
    'low_cost': 200*K, # low end on cost of intervention
    'high_cost': 800*K, # high end on cost of intervention
    'low_yrs_credit': 1, # low end on years of credit for intervention
    'high_yrs_credit': 5, # high end on years of credit for intervention
    }

# The duration in hours of DALY-equivalent harm caused by 
#     each welfare threat over the 
#     typical shrimp's life. This is based on Hannah McKay
#     and William McAuliffe's shrimp pain track estimates, 
#     available under the Modified_Shrimp_Welfare_Prioritization.Rmd file.

harm_duration_dict = {'high_density': sq.lognorm(0.820, 197.7), 
                      'ammonia': sq.lognorm(0.382, 122.0),
                      'lack_substrate': sq.lognorm(0.249, 56.8),
                      'low_dissolved_oxygen': sq.lognorm(0.101, 39.6),
                      'low_salinity': sq.lognorm(0.166, 44.35),
                      'water_based_transit': sq.lognorm(0.194, 4.82),
                      'ph': sq.lognorm(0.074, 24.0),
                      'underfeeding': sq.lognorm(0.034, 19.4),
                      'high_temp': sq.lognorm(0.035, 15.0),
                      'water_pollution': sq.lognorm(0.0127, 9.02),
                      'low_temp': sq.lognorm(0.00583, 4.75),
                      'malnutrition': sq.lognorm(0.0035, 3.61),
                      'harvest': sq.lognorm(0.0748, 1.05),
                      'predators': sq.lognorm(0.00206, 1.43),
                      'slaughter': sq.lognorm(0.0505, 0.384),
                      'waterless_transit': sq.lognorm(0.0104, 0.110),
                      'eyestalk_ablation': sq.lognorm(1.5*10**-10, 4*10**-5),
                }

# The approximate number of shrimp who die on farms annually
# based on a report by Daniela Waldhorn at Rethink Priorities. 
num_shrimp_annually = sq.sample(sq.lognorm(650*B, 1400*B), N)


def get_human_daly_burden_from_harm_type(intervention_dict, sc_welfare_range = None, to_print=False):
    '''
    The number of sentience-conditioned, human-equivalent DALYs per year 
        that shrimp experience from welfare threat(s) that can be addressed by a particular intervention.

    '''
    harm_types_lst = intervention_dict['harm_types']
    total_annual_shrimp_daly_burden_by_harms = np.zeros(N)

    for harm_type in harm_types_lst:
        per_shrimp_hrs_harm_duration = sq.sample(harm_duration_dict[harm_type], N)
        per_shrimp_yrs_harm_duration = per_shrimp_hrs_harm_duration / 24 / 365
        total_annual_shrimp_daly_burden_by_harms += per_shrimp_yrs_harm_duration * num_shrimp_annually

    if sc_welfare_range == None:
        sent_conditioned_wr_shrimp = shrimp_sentience_conditioned_welfare_range()
    else: 
        sent_conditioned_wr_shrimp = sq.sample(sq.uniform(sc_welfare_range[0], sc_welfare_range[1]), N)
    sent_conditioned_annual_human_daly_burden = total_annual_shrimp_daly_burden_by_harms * sent_conditioned_wr_shrimp

    speed_up_years_low = intervention_dict['low_yrs_credit']
    speed_up_years_high = intervention_dict['high_yrs_credit']

    speed_up_years = sq.sample(sq.lognorm(speed_up_years_low, speed_up_years_high, lclip=1, rclip=20), N)

    total_sent_conditioned_human_daly_burden = speed_up_years*sent_conditioned_annual_human_daly_burden
    if to_print:
        print("Mean sentience-conditioned annual human-DALY burden: {}".format(np.mean(sent_conditioned_annual_human_daly_burden)))
        print("Percentiles: ")
        print(sq.get_percentiles(sent_conditioned_annual_human_daly_burden))
        print("Mean shrimp-daly burden by harms: {}".format(np.mean(total_annual_shrimp_daly_burden_by_harms)))
        print("Percentiles: ")
        print(sq.get_percentiles(total_annual_shrimp_daly_burden_by_harms))
        print("Mean total sentience-conditioned human DALY burden over 2-10 year speed up: {}".format(np.mean(total_sent_conditioned_human_daly_burden)))
        print("Percentiles: ")
        print(sq.get_percentiles(total_sent_conditioned_human_daly_burden))

    return total_sent_conditioned_human_daly_burden

def get_years_credit(intervention_dict):
    '''
    The number of years of credit for a given intervention, based on the intervention dictionary.
    '''
    years_credit = sq.sample(sq.lognorm(intervention_dict['low_yrs_credit'],
                                        intervention_dict['high_yrs_credit'],
                                        lclip=0.1*intervention_dict['low_yrs_credit'],
                                        rclip=10*intervention_dict['high_yrs_credit']), N)
    return years_credit

def get_shrimp_helped_per_dollar(intervention_dict):
    '''
    Get the number of shrimp helped per dollar spent on a given intervention, based on the intervention dictionary.
    '''
    prop_shrimp_affected = sq.sample(sq.lognorm(intervention_dict['low_p_shrimp_affected'], 
                                                       intervention_dict['high_p_shrimp_affected'], 
                                                       lclip=0.1*intervention_dict['low_p_shrimp_affected'], 
                                                      rclip=10*intervention_dict['high_p_shrimp_affected']), N)
    num_affected = num_shrimp_annually * prop_shrimp_affected 

    years_credit = get_years_credit(intervention_dict)

    prob_success = sq.sample(sq.lognorm(intervention_dict['low_p_success'],
                                        intervention_dict['high_p_success'],
                                        lclip=0.1*intervention_dict['low_p_success'],
                                        rclip=1), N)
    
    cost_intervention = sq.sample(sq.lognorm(intervention_dict['low_cost'],
                                                intervention_dict['high_cost'],
                                                lclip=0.1*intervention_dict['low_cost'],
                                                rclip=5*intervention_dict['high_cost']), N)
    
    shrimp_helped_per_dollar = num_affected * years_credit * prob_success / cost_intervention

    return shrimp_helped_per_dollar

def generic_sentience_conditioned_shrimp_dalys_per_1000(intervention_dict, to_print=False):
    '''
    For a generic shrimp project (details provided by the intervention dictionary), 
        this estimates the number of shrimp-DALYs averted per $1000 spent on the 
        intervention. The variables include: number of shrimp in the world experiencing the 
        welfare threat, proportion of global shrimp affected by a given campaign, the duration 
        of disabling pain for the welfare threat, the proportion of harm reduced by the intervention,
        the probability of success of the intervention, and the cost of the intervention.
    This is NOT adjusted for p(sentience) and welfare ranges.
    '''
    shrimp_helped_per_dollar = get_shrimp_helped_per_dollar(intervention_dict)

    harm_types_lst = intervention_dict['harm_types']
    dalys_reduced_per_shrimp = np.zeros(N)

    for harm_type in harm_types_lst:
        hrs_harm_duration = sq.sample(harm_duration_dict[harm_type], N)
        yrs_harm_duration = hrs_harm_duration / 24 / 365

        low_prop_harm_reduced = intervention_dict['low_suffering_reduced_proportion'][harm_type]
        high_prop_harm_reduced = intervention_dict['high_suffering_reduced_proportion'][harm_type]
        prop_harm_reduced_if_good = sq.lognorm(low_prop_harm_reduced, high_prop_harm_reduced,
                                                lclip=0.1*low_prop_harm_reduced,
                                                rclip=1)
        prop_harm_reduced_if_bad = -0.5*prop_harm_reduced_if_good
        
        prop_harm_reduced = sq.sample(sq.mixture([prop_harm_reduced_if_bad, prop_harm_reduced_if_good], [0.03, 0.97]), N)
        dalys_reduced_per_shrimp += yrs_harm_duration * prop_harm_reduced 
    
    shrimp_dalys_per_dollar = dalys_reduced_per_shrimp * shrimp_helped_per_dollar
    
    shrimp_dalys_per_1000 =  shrimp_dalys_per_dollar * 1000

    if to_print:
        print("Mean number of shrimp helped per dollar: {}".format(np.mean(shrimp_helped_per_dollar)))
        print("Percentiles: ")
        print(sq.get_percentiles(shrimp_helped_per_dollar))
        print(f'Mean sentience-conditioned Shrimp-DALYs per $1000: {np.mean(shrimp_dalys_per_1000)}')
        print("Percentiles:")
        print(sq.get_percentiles(shrimp_dalys_per_1000))
    return shrimp_dalys_per_1000

def sample_is_shrimp_sentient():
    '''
    Create a vector of binary variables representing whether a shrimp is sentient,
        based on our subjective 90% CI on the probability that shrimp are sentient.
    '''
    p_sent_low = 0.2
    p_sent_high = 0.7
    p_sent_lclip = 0.01
    p_sent_rclip = 1
    
    p_sent = sq.sample(sq.lognorm(p_sent_low, p_sent_high, lclip = p_sent_lclip, rclip=p_sent_rclip), N)
    binaries_is_sent = np.zeros(N)

    for i in range(N):
        X = np.random.binomial(1, p_sent[i])
        binaries_is_sent[i] = X

    return binaries_is_sent

def shrimp_sentience_conditioned_welfare_range(to_print=False):
    '''
    Create a vector of welfare ranges for shrimp, conditional on being sentient.
        Based on RP's moral weight project: 
        https://docs.google.com/spreadsheets/d/1gJZlOTmrWwR6C7us5G0-aRM9miFeEcP11_6HEfpCPus/edit?usp=sharing
    '''
    wr_shrimp_lower = 0.01
    wr_shrimp_upper = 2 # a bit less than the 95th percentile so the mean would match 0.439
    wr_shrimp_lclip = 0.000001 # neuron count for a shrimp
    wr_shrimp_rclip = 5 # a bit less than the 95th percentile for the undiluted experiences model

    wr_shrimp = sq.sample(sq.lognorm(wr_shrimp_lower, wr_shrimp_upper, lclip=wr_shrimp_lclip, rclip=wr_shrimp_rclip), N)

    if to_print:
        print(f'Mean sentience-conditioned welfare range for shrimp: {np.mean(wr_shrimp)}')
        print("Percentiles:")
        print(sq.get_percentiles(wr_shrimp))

    return wr_shrimp

def shrimp_sentience_conditioned_human_dalys_per_1000(intervention_dict, sc_welfare_range=None, to_print=False):
    '''
    Conditional on shrimp sentience, the number of human-equivalent DALYs averted per $1000 spent on the intervention.
    '''
    sentience_conditioned_shrimp_dalys_per_1000 = generic_sentience_conditioned_shrimp_dalys_per_1000(intervention_dict, to_print=to_print)

    if sc_welfare_range == None:
        sc_wr_shrimp = shrimp_sentience_conditioned_welfare_range()
        sentience_conditioned_human_dalys_per_1000 = sentience_conditioned_shrimp_dalys_per_1000 * sc_wr_shrimp

    else:
        lower = sc_welfare_range[0]
        upper = sc_welfare_range[1]
        sc_wr_shrimp = sq.sample(sq.uniform(lower, upper), N)
        sentience_conditioned_human_dalys_per_1000 = sentience_conditioned_shrimp_dalys_per_1000 * sc_wr_shrimp

    if to_print:
        print(f'Mean sentience-conditioned welfare range for shrimp: {np.mean(sc_wr_shrimp)}')
        print(f'Mean sentience-conditioned human-DALYs per $1000: {np.mean(sentience_conditioned_human_dalys_per_1000)}')
        print("percentiles")
        print(sq.get_percentiles(sentience_conditioned_human_dalys_per_1000))

    return sentience_conditioned_human_dalys_per_1000

def shrimp_campaign_human_dalys_per_1000(intervention_dict, sc_welfare_range=None, to_print=False):
    '''
    Take the sentience-conditioned human-equivelent DALYs averted per $1000 spent on the intervention, whether
        a shrimp is sentient, and the welfare range for shrimp, and return the human-equivalent DALYs averted per $1000
        spent on the type of shrimp welfare project specified. 
    '''
    if sc_welfare_range == None:
        is_sent = sample_is_shrimp_sentient()
        sentience_conditioned_human_dalys_per_1000_shrimp = shrimp_sentience_conditioned_human_dalys_per_1000(intervention_dict)
        human_daly_equivalent_dalys_per_1000_shrimp_campaign = sentience_conditioned_human_dalys_per_1000_shrimp * is_sent
    else:
        is_sent = sample_is_shrimp_sentient()
        sc_human_daly_equivalent_dalys_per_1000_shrimp_campaign = \
            shrimp_sentience_conditioned_human_dalys_per_1000(intervention_dict, sc_welfare_range=sc_welfare_range)
        human_daly_equivalent_dalys_per_1000_shrimp_campaign = sc_human_daly_equivalent_dalys_per_1000_shrimp_campaign * is_sent
    if to_print:
        print(f'Mean human-DALYs per 1000 shrimp: {np.mean(human_daly_equivalent_dalys_per_1000_shrimp_campaign)}')
        print("Percentiles:")
        print(sq.get_percentiles(human_daly_equivalent_dalys_per_1000_shrimp_campaign))

    return human_daly_equivalent_dalys_per_1000_shrimp_campaign
